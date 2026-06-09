
import os
import streamlit as st
from modules.stl_module import calc_stl, plot_stl_components
from modules.data_loader import create_timeseries

def generate_html_report_full(
    substance: str,
    fig_raw,                    
    fig_stl,
    fig1,
    fig2,
    fig_hist,
    fig_comp,
    classical_fig,
    classical_table,
    ti_result,                 # st.session_state[key_ti] をそのまま渡す
    ti_text: str,
    robustness_score_1: float,
    robustness_score_2: float,
    gap_s2: float,
    w2: float,
    rmse_param_2: float,
    gap2: float,
    width_s2: float,
    tau_penalty_2: float,
    total_score_1: float,
    total_score_2: float,
    report_data: dict,
    breakpoint_table_html: str,

    output_dir: str = "output",
):
    """
    Tab4 で作成した解析結果（STL / Posterior 50 draws / τ 分布 / Posterior Mean Fit / TI /
    判定見解ロジック / 堅牢性評価テーブル / 変化点ウィンドウ）を
    1 つの HTML レポートとして output/ 以下に保存する関数。
    """

    import os
    import re
    from datetime import datetime
    #import datetime
    import matplotlib.pyplot as plt

    # -----------------------------
    # 0. 出力ディレクトリ準備
    # -----------------------------
    out_dir = output_dir
    plot_dir = os.path.join(out_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    

    # -----------------------------
    # 1. 図の保存
    # -----------------------------
    plot_file_name0 = f"{substance}_fig_raw_{timestamp}.png"
    #plot_file_name_pwlf = f"{substance}_fig_pwlf_{timestamp}.png"
    plot_file_name1 = f"{substance}_fig_stl_{timestamp}.png"
    plot_file_name2 = f"{substance}_fig1_{timestamp}.png"
    plot_file_name3 = f"{substance}_fig2_{timestamp}.png"
    plot_file_name4 = f"{substance}_fig_hist_{timestamp}.png"
    plot_file_name_mean = f"{substance}_mean_fit_{timestamp}.png"

    plot_file_path0 = os.path.join(plot_dir, plot_file_name0)
    #plot_file_path_pwlf = os.path.join(plot_dir, plot_file_name_pwlf)
    plot_file_path1 = os.path.join(plot_dir, plot_file_name1)
    plot_file_path2 = os.path.join(plot_dir, plot_file_name2)
    plot_file_path3 = os.path.join(plot_dir, plot_file_name3)
    plot_file_path4 = os.path.join(plot_dir, plot_file_name4)
    plot_file_path_mean = os.path.join(plot_dir, plot_file_name_mean)

    # --- 1-1. STL分解プロットの保存 ---
    if fig_raw is not None:
        fig_raw.savefig(plot_file_path0, dpi=300, bbox_inches="tight")
        html_img_raw = f'<div class="img-container"><img src="plot/{plot_file_name0}" alt="raw data plot"></div>'
    else:    
        html_img_raw = ""

    if fig_stl is not None:
        fig_stl.savefig(plot_file_path1, dpi=300, bbox_inches="tight")
        html_img_stl = f'<img src="plot/{plot_file_name1}" alt="STL Decomposition">'
    else:
 
        html_img_stl = "<p>STL analysis was not performed.</p>"


    # --- 1-6. Classical Breakpoint Plot 保存 ---
    if classical_fig is not None:
        plot_file_name_classical = f"{substance}_classical_{timestamp}.png"
        plot_file_path_classical = os.path.join(plot_dir, plot_file_name_classical)
        classical_fig.savefig(plot_file_path_classical, dpi=300, bbox_inches="tight")
        html_img_classical = f'<div class="img-container"><img src="plot/{plot_file_name_classical}" alt="Classical Breakpoint Plot"></div>'
    else:
        html_img_classical = "<p>Classical analysis was not performed for this substance.</p>"

    # --- 1-2. 1-break model プロットの保存 ---
    if fig1 is not None:
        fig1.savefig(plot_file_path2, dpi=300, bbox_inches="tight")
        html_img_fig1 = f'<img src="plot/{plot_file_name2}" alt="1-break model">'
    else:
        html_img_fig1 = "<p>1-break model plot not available.</p>"

    # --- 1-3. 2-break model プロットの保存 ---
    if fig2 is not None:
        fig2.savefig(plot_file_path3, dpi=300, bbox_inches="tight")
        html_img_fig2 = f'<img src="plot/{plot_file_name3}" alt="2-break model">'
    else:
        html_img_fig2 = "<p>2-break model plot not available.</p>"

    # --- 1-4. 変化点分布プロットの保存 ---
    if fig_hist is not None:    
        fig_hist.savefig(plot_file_path4, dpi=300, bbox_inches="tight")
        html_img_hist = f'<img src="plot/{plot_file_name4}" alt="Estimated Breakpoint Distributions">'
    else:
        html_img_hist = "<p>Breakpoint distribution plot not available.</p>"

    # --- 1-5. 代表線フィットプロットの保存 ---
    if fig_comp is not None:        
        fig_comp.savefig(plot_file_path_mean, dpi=300, bbox_inches="tight")
        html_img_mean = f'<img src="plot/{plot_file_name_mean}" alt="Posterior Mean Parameter Fit">'
    else:
        html_img_mean = "<p>Mean parameter fit plot not available.</p>"

    #Classical の区間統計テーブルを HTML に変換

    if classical_table is not None:
        classical_table_html = classical_table.to_html(
            index=False,
            border=1,
            justify="center",
            classes="classical-table"
        )
    else:
        classical_table_html = "<p>No classical breakpoint statistics available.</p>"
    # -----------------------------
    # if is_short_mode:モードによる分岐
    # -----------------------------
    # report_data が空、または必要なキーがない場合は「簡易版」として処理
    is_short_mode = (not report_data or "評価指標 (Metrics)" not in report_data)
    #    ⚠️ 簡易版用のテキストを差し込んで終了する           薄すぎる黄色     .warning-box {{ background-color: #fef9e7; border-color: #f1c40f; }}
    if is_short_mode:
        file_name = f"{substance}_bayesian_report_{timestamp}.html"
        file_path = os.path.join(out_dir, file_name)
        html_content = f"""<!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>POPs Trend Analysis Report - {substance}</title>
            <style>
                body {{ font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #34495e; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                .img-container {{ text-align: center; margin: 20px 0; }}
                .img-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 2px 2px 12px rgba(0,0,0,0.1); }}
                .box {{ padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 5px solid; }}
                .info-box {{ background-color: #e8f4f8; border-color: #2980b9; }}
                .success-box {{ background-color: #eafaf1; border-color: #2ecc71; }}
                .warning-box {{background-color: #fff9db; border-color: #fab005; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th {{ background-color: #34495e; color: white; padding: 12px; border: 1px solid #ddd; }}
                .meta {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>{substance}: POPs Monitoring Trend Analysis Report</h1>
            <p class="meta">対象物質: <strong>{substance}</strong> | 出力日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
                <p>このレポートは「簡易版モード」で生成されており、ベイジアン分析の詳細な評価指標やTIの結果は含まれていません。</p>
                        {html_img_raw}
                        <p>>Raw data plot was not performed.</p>
            <div class="img-container">
                <h2>📉 Classical Breakpoint Analysis</h2>
                <h3>● Classical Breakpoint Plot</h3>

                <div class="img-container">
                    {html_img_classical}
                
                </div>
            
                <p>Classical analysis was not performed for this substance.</p>
            

                <h3>● Interval Regression Statistics</h3>
                {classical_table_html}



                <h2>📊 STL Decomposition: {substance}</h2>
                <p>STL 分解プロットは 4 段構成（Observed / Trend / Seasonal / Residual）です</p>
                {html_img_stl}

                <br><br>

        """
        html_content += "<p style='color:red;'>※Bayesian計算およびTIは未実行です。</p>"


        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return # 簡易版はここで終了





    # --- 1-7. TI 図の生成（ti_result があれば） ---
    html_img_ti = "<p>TI は実行されていません。</p>"
    if ti_result is not None:
        fig_ti, ax_ti = plt.subplots(figsize=(10, 4.5))
        ax_ti.plot(ti_result["beta"], ti_result["E1"], "-o", ms=3, label="1-break")
        ax_ti.plot(ti_result["beta"], ti_result["E2"], "-o", ms=3, label="2-break")
        ax_ti.set_title("Thermodynamic Integration (TI) Curve")
        ax_ti.set_xlabel("β")
        ax_ti.set_ylabel("E[log likelihood]")
        ax_ti.legend()

        plot_file_name5 = f"{substance}_fig_ti_{timestamp}.png"
        plot_file_path5 = os.path.join(plot_dir, plot_file_name5)
        fig_ti.savefig(plot_file_path5, dpi=300, bbox_inches="tight")
        plt.close(fig_ti) # メモリ解放のためクローズ

        html_img_ti = f'<img src="plot/{plot_file_name5}" alt="Thermodynamic Integration Curve">'


    # -----------------------------
    # 2. HTML 用のパスとテキスト整形
    # -----------------------------
    html_img_path1 = f"plot/{plot_file_name1}"
    html_img_path2 = f"plot/{plot_file_name2}"
    html_img_path3 = f"plot/{plot_file_name3}"
    html_img_path4 = f"plot/{plot_file_name4}"
    html_img_path_mean = f"plot/{plot_file_name_mean}"

    # Markdown の **太字** を <b> に変換

    ti_html = "<p>" + re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", ti_text) + "</p>"

    # -----------------------------
    # 3. 判定見解テキスト（元コードをそのまま関数内に移植）
    # -----------------------------
    if robustness_score_1 is not None:
        if robustness_score_1 < robustness_score_2 and gap_s2 > 1:
            recommendation_title = "💡 総合評価に基づく推奨: 「1-break モデル」の採用検討"
            recommendation_text = f"""
            
            <p><strong>【判断のロジック】</strong><br>
            形式的な情報量基準（WAIC）では 2-break モデルが圧倒的（確率 {w2*100:.1f}%）に支持されています。
            しかし、モデルの構造的安定性を詳細に検証すると、2-break側で以下のリスク（ペナルティ）が確認されました。</p>
            <ol>
                <li><strong>代表線の適合不全</strong>: 代表線フィット曲線におけるデータとの乖離（RMSE: {rmse_param_2:.4f}）が1-breakより劣ります。</li>
                <li><strong>大きな非線形ギャップ</strong>: 事後予測平均 E[μ] と代表線 μ(E[θ]) との間に乖離（Gap: {gap2:.4f}）が生じており、過学習の兆候が見られます。</li>
                <li><strong>変化点 (break) の不安定性</strong>: 変化点の信用区間幅ペナルティ: {width_s2:.2f}）、または、分離安定性（ペナルティ: {tau_penalty_2:.2f}）に課題があります。</li>
            </ol>
            {ti_html}

            """
        elif gap_s2 < 1 and robustness_score_2 < robustness_score_1:
            recommendation_title = "🏆 総合評価に基づく推奨: 「2-break モデル」の採用"
            recommendation_text = f"""
            <p><strong>【判断のロジック】</strong><br>
            統計的情報量基準（WAIC）の強い支持に加え、多角的な堅牢性評価における総合ペナルティスコアでも
            2-break モデル（スコア: {total_score_2:.2f}）が 1-break モデル（スコア: {total_score_1:.2f}）を上回る（ペナルティが小さい）結果となっています。<br>
            代表線の適合性、非線形ギャップの小ささ、変化点分布の分離安定性がすべて健全であると判断できるならば、構造変化が2段階で起きていると解釈するのが最も妥当です。
            <ol>
                <li><strong>代表線の適合不全</strong>: 代表線フィット曲線におけるデータとの乖離（RMSE: {rmse_param_2:.4f}）が1-breakより劣ります。</li>
                <li><strong>大きな非線形ギャップ</strong>: 事後予測平均 E[μ] と代表線 μ(E[θ]) との間に乖離（Gap: {gap2:.4f}）が生じており、過学習の兆候が見られます。</li>
                <li><strong>変化点 (break) の不安定性</strong>: 変化点の信用区間幅ペナルティ: {width_s2:.2f}）、または、分離安定性（ペナルティ: {tau_penalty_2:.2f}）に課題があります。</li>
            </ol>
            {ti_html}

            """
        else:
            recommendation_title = "⚠️ 総合評価に基づく推奨: 要慎重判断（シンプルなモデルを推奨）"
            recommendation_text = f"""
            <p><strong>【判断のロジック】</strong><br>
            総合ペナルティスコアは 1-break ({total_score_1:.2f}) vs 2-break ({total_score_2:.2f}) となり、両者の性能に一長一短があるが、多角的な堅牢性ペナルティスコアに注意が必要である。
            統計モデルの原則（オッカムの剃刀）に従い、過剰なパラメータ化を避けるため、よりシンプルで頑健な<strong>「1-break モデル」</strong>をベースに考察を進めることを推奨します。</p>
            <ol>
                <li><strong>代表線の適合不全</strong>: 代表線フィット曲線におけるデータとの乖離（RMSE: {rmse_param_2:.4f}）が1-breakより劣ります。</li>
                <li><strong>大きな非線形ギャップ</strong>: 事後予測平均 E[μ] と代表線 μ(E[θ]) との間に乖離（Gap: {gap2:.4f}）が生じており、過学習の兆候が見られます。</li>
                <li><strong>変化点 (break) の不安定性</strong>: 変化点の信用区間幅ペナルティ: {width_s2:.2f}）、または、分離安定性（ペナルティ: {tau_penalty_2:.2f}）に課題があります。</li>
            </ol>
            {ti_html}

            """

    # -----------------------------
    # 4. 堅牢性テーブル HTML
    # -----------------------------

    table_rows_html = ""
    if not is_short_mode:
        for idx, metric in enumerate(report_data["評価指標 (Metrics)"]):
            v1 = report_data["1-break Model"][idx]
            v2 = report_data["2-break Model"][idx]
            style = (
                "font-weight: bold; background-color: #f9f9f9;"
                if idx == len(report_data["評価指標 (Metrics)"]) - 1
                else ""
            )
            table_rows_html += f"""
            <tr style="{style}">
                <td style="padding: 10px; border: 1px solid #ddd;">{metric}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{v1}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{v2}</td>
            </tr>
            """

            #Classical の区間統計テーブルを HTML に変換

            if classical_table is not None:
                classical_table_html = classical_table.to_html(
                    index=False,
                    border=1,
                    justify="center",
                    classes="classical-table"
                )
            else:
                classical_table_html = "<p>No classical breakpoint statistics available.</p>"

    # -----------------------------
    # 5. HTML 全体テンプレート
    # -----------------------------
    file_name = f"{substance}_bayesian_report_{timestamp}.html"
    file_path = os.path.join(out_dir, file_name)

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>POPs Trend Analysis Report - {substance}</title>
    <style>
        body {{ font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #34495e; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        .img-container {{ text-align: center; margin: 20px 0; }}
        .img-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 2px 2px 12px rgba(0,0,0,0.1); }}
        .box {{ padding: 20px; border-radius: 6px; margin: 20px 0; border-left: 5px solid; }}
        .info-box {{ background-color: #e8f4f8; border-color: #2980b9; }}
        .success-box {{ background-color: #eafaf1; border-color: #2ecc71; }}
        .warning-box {{ background-color: #eafaf1; border-color: #2ecc71; }}
        /* .warning-box {{ background-color: #fff9db; border-color: #fab005; }} */
         /* テーブル全体の統一スタイル */
        table {{
            width: 100%;
            border-collapse: collapse; /* 罫線を二重線にせず1本にまとめる */
            margin: 20px 0;
            font-family: sans-serif;
        }}

        /* ヘッダーのスタイル（濃い青の背景に白文字） */
        table th {{
            background-color: #1a446c; /* 画面に合わせた濃いめのネイビーブルー */
            color: #ffffff;
            font-weight: bold;
            padding: 10px 12px;
            text-align: left;
            border: 1px solid #dcdcdc; /* 外枠・内枠の薄いグレーの罫線 */
        }}

        /* データ行のスタイル */
        table td {{
            padding: 10px 12px;
            border: 1px solid #dcdcdc; /* スッキリとした格子状の薄い罫線 */
            color: #333333;
            background-color: #ffffff;
        }}

        /* （オプション）行を交互に色分けしてさらに見やすくする場合 */
        table tr:nth-child(even) td {{
            background-color: #f9f9f9;
        }}




        .meta {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{substance}: POPs Monitoring Trend Analysis Report</h1>
    <p class="meta">対象物質: <strong>{substance}</strong> | 出力日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="img-container">
        <h2>📉 Classical Breakpoint Analysis</h2>
        <h3>● Classical Breakpoint Plot</h3>

        <div class="img-container">
            {html_img_classical}
           
        </div>
      

       

        <h3>● Interval Regression Statistics</h3>
        {classical_table_html}



        <h2>📊 STL Decomposition: {substance}</h2>
        <p>STL 分解プロットは 4 段構成（Observed / Trend / Seasonal / Residual）です</p>
        {html_img_stl}

        <br><br>
        <h2>⏱️ Mean of Posterior Fitted Values E[μ]:</h2>
        <p>1-break model</p>
        {html_img_fig1}
        
        <p>2-break model</p>
        {html_img_fig2}
        

        <h2>⏱️ Estimated Breakpoint Distributions</h2>
        {html_img_hist}


        <h2>📊 Posterior Mean Parameter Fit — μ(E[θ])</h2>
        <p>事後分布の平均パラメータ（α, β, τ）を用いて計算した代表的なフィット曲線です。</p>
        {html_img_mean}


        <h2>⏱️ Thermodynamic Integration Curve (β vs Expected Log-Likelihood)</h2>
        <p>逆温度 β の変化に伴う期待対数尤度の推移です。この曲線の下側の面積の差が、モデルの優劣（周辺尤度の差）になります。</p>
        {html_img_ti}
    </div>
    <h2>📌 モデル採用に関する見解</h2>
    <div class="box {{% if '💡' in {recommendation_title} %}}info-box
    {{% elif '🏆' in {recommendation_title} %}}success-box
    {{% else %}}warning-box{{% endif %}}">
        <h3>{recommendation_title}</h3>
            {recommendation_text}
    </div>

    <h2>📉 Bayesian Breakpoint Results</h2>

    {breakpoint_table_html}

    #<h2>🛡️ 多角的なモデル堅牢性の評価</h2>
    #<table>
    #    <thead>
    #        <tr>
    #            <th>評価指標 (Metrics)</th>
    #            <th>1-break Model</th>
    #            <th>2-break Model</th>
    #        </tr>
    #   </thead>
    #    <tbody>
    #       {table_rows_html}
    #    </tbody>
    #</table>
</body>
</html>
"""

    # -----------------------------
    # 6. ファイル書き込み
    # -----------------------------
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return file_path







