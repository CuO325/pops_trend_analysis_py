import base64
import mimetypes
from pathlib import Path
from bs4 import BeautifulSoup


def embed_images_in_html(input_html_path, output_html_path):
    input_path = Path(input_html_path)
    base_dir = input_path.parent  # 画像相対パスの基準ディレクトリ

    # 1. HTMLファイルの読み込み
    with open(input_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # 2. すべての <img> タグを処理
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue

        # すでにBase64化されている場合やWeb URL(http/https)はスキップ
        if src.startswith("data:") or src.startswith(("http://", "https://")):
            continue

        # 画像ファイルのパス解決
        img_path = (base_dir / src).resolve()

        if img_path.exists() and img_path.is_file():
            # MIMEタイプの取得（例: image/png, image/jpeg）
            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type:
                mime_type = "image/png"  # 判定できない場合のデフォルト

            # 画像データをBase64にエンコード
            with open(img_path, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode(
                    "utf-8"
                )

            # src属性を Base64 データURI に置き換え
            img.attrs["src"] = f"data:{mime_type};base64,{encoded_string}"
            print(f"埋め込み成功: {src}")
        else:
            print(f"警告: 画像ファイルが見つかりません -> {img_path}")

    # 3. 変換後のHTMLを出力
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(str(soup))

    print(f"\n変換完了: {output_html_path}")


# 実行
if __name__ == "__main__":
    # 対象のHTMLファイルパスと出力先を指定してください
    input_html = r"C:\GitHub\pops_trend_analysis_py\ST_BREAK_POINTs\examples\HCBD(Hexachlorobutadiene)_bayesian_report_20260808_103101.html"
    output_html =  r"C:\GitHub\pops_trend_analysis_py\ST_BREAK_POINTs\examples\HCBD(Hexachlorobutadiene)_bayesian_report.html"

    embed_images_in_html(input_html, output_html)