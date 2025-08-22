import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


class SECDownloader:
    # download sec primary html docs from csv (arg: str csv_path, str out_dir, str user_agent; output: dict summary)
    def __init__(self, csv_path: str, out_dir: str, user_agent: str = "contact email required"):
        self.csv_path = csv_path
        self.out_dir = Path(out_dir)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        })

    # ensure output dir exists (arg: none; output: none)
    def _ensure_dir(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # load and normalize csv (arg: none; output: pd dataframe)
    def _load_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "rdate" in df.columns:
            df["rdate"] = pd.to_datetime(df["rdate"], errors="coerce")
        if "cik" in df.columns:
            df["cik"] = df["cik"].astype(str).str.zfill(10)

        link_col = "annualreport_link" if "annualreport_link" in df.columns else ("link" if "link" in df.columns else None)
        if link_col is None:
            raise ValueError("csv must have 'annualreport_link' or 'link'")
        if link_col != "annualreport_link":
            df["annualreport_link"] = df[link_col]

        need = {"cik", "rdate", "annualreport_link"}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"csv missing: {missing}")
        return df

    # build filename from url+cik+date (arg: str url, str cik, pd timestamp date; output: str filename)
    def _filename(self, url: str, cik: str, dt) -> str | None:
        m = re.search(r"/(\d{10}-\d{2}-\d{6})", url)
        if m:
            accession = m.group(1)
        else:
            m = re.search(r"/(\d{18})", url)
            if not m:
                return None
            raw = m.group(1)
            accession = f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"
        doc_m = re.search(r"/([^/]+\.(?:htm|html))$", url, re.IGNORECASE)
        doc = doc_m.group(1) if doc_m else "document.htm"
        year_end = f"{int(dt.year)}1231" if pd.notna(dt) else "unknown_date"
        return f"{cik}-{year_end}-{accession}-{doc}"

    # find primary html from index page (arg: str index_url; output: str url or none)
    def _primary_html_url(self, index_url: str) -> str | None:
        try:
            time.sleep(0.12)
            r = self.session.get(index_url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            table = soup.find("table", class_="tableFile", summary="Document Format Files")
            if not table:
                return None
            for row in table.find_all("tr")[1:]:
                tds = row.find_all("td")
                if len(tds) < 4:
                    continue
                doc_type = (tds[3].get_text(strip=True) or "").lower()
                link = tds[2].find("a")
                href = link["href"] if link and link.has_attr("href") else None
                if not href:
                    continue
                if "10-k" in doc_type or "primary document" in doc_type:
                    return urljoin(index_url, href)
            # fallback to first html link
            for row in table.find_all("tr")[1:]:
                link = row.find("a")
                href = link["href"] if link and link.has_attr("href") else None
                if href and re.search(r"\.(htm|html)$", href, re.I):
                    return urljoin(index_url, href)
            return None
        except requests.RequestException:
            return None

    # write url to path with retries (arg: str url, path save_path, int retries; output: bool ok)
    def _download(self, url: str, save_path: Path, retries: int = 3) -> bool:
        for _ in range(retries):
            try:
                time.sleep(0.12)
                r = self.session.get(url, timeout=30)
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "").lower()
                if "text/html" not in ctype:
                    return False
                save_path.write_bytes(r.content)
                return self._valid_html(save_path)
            except requests.RequestException:
                time.sleep(1)
        return False

    # basic html validation (arg: path save_path; output: bool ok)
    def _valid_html(self, save_path: Path) -> bool:
        try:
            txt = save_path.read_text(encoding="utf-8", errors="ignore").lower()
            return "<html" in txt and "</html>" in txt
        except Exception:
            return False

    # run download pipeline (arg: none; output: dict summary)
    def run(self) -> dict:
        self._ensure_dir()
        df = self._load_df()
        succ = fail = skip = 0
        for _, row in df.iterrows():
            index_url = row["annualreport_link"]
            if pd.isna(index_url):
                fail += 1
                continue
            primary = self._primary_html_url(str(index_url))
            if not primary:
                fail += 1
                continue
            fname = self._filename(primary, str(row["cik"]), row["rdate"])
            if not fname:
                fail += 1
                continue
            out = self.out_dir / fname
            if out.exists():
                skip += 1
                continue
            if self._download(primary, out):
                succ += 1
            else:
                fail += 1
        return {"total": len(df), "successes": succ, "failures": fail, "skipped": skip}


def main():
    # cli entry (arg: env or argv; output: none)
    import argparse
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("csv_path", help="path to csv")
    p.add_argument("out_dir", help="output dir")
    p.add_argument("--user-agent", default="contact email required", help="sec user agent")
    args = p.parse_args()

    dl = SECDownloader(args.csv_path, args.out_dir, args.user_agent)
    res = dl.run()
    print(res)


if __name__ == "__main__":
    main()
