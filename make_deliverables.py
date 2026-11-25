"""Build the report and slides directly from the study's aggregate outputs."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = "#172A38"
MUTED = "#425A68"
TEAL = "#006E66"
PALE = "#E8F4F0"
GOLD = "#B96438"
GRAY = "#82949E"
WHITE = "#FFFFFF"
MODELS = ["Median", "Ridge", "Random forest"]
MODEL_COLORS = [GRAY, GOLD, TEAL]
REFERENCES = [
    ("NJ Division of Taxation. Statistical Information (sales and property assessment archives).",
     "https://www.nj.gov/treasury/taxation/lpt/statdata.shtml"),
    ("NJ Division of Taxation. SR-1A fixed-width file layout.",
     "https://www.nj.gov/treasury/taxation/pdf/lpt/SR1Afilelayout.pdf"),
    ("NJ Division of Taxation. MOD-IV fixed-width file layout and district codes.",
     "https://www.nj.gov/treasury/taxation/pdf/lpt/modivlayout.pdf"),
    ("Rutgers University. N.J. MOD IV Historical Database.",
     "https://modiv.rutgers.edu/about/"),
    ("Zillow. What is a Zestimate?",
     "https://www.zillow.com/zestimate/"),
    ("Sirmans, Macpherson & Zietz (2005). The Composition of Hedonic Pricing Models. Journal of Real Estate Literature, 13(1), 1–44.",
     "https://doi.org/10.1080/10835547.2005.12090154"),
    ("Antipov & Pokryshevskaya (2012). Mass appraisal of residential apartments. Expert Systems with Applications, 39(2), 1772–1778.",
     "https://doi.org/10.1016/j.eswa.2011.08.077"),
    ("Čeh, Kilibarda, Lisec & Bajat (2018). Estimating the Performance of Random Forest versus Multiple Regression for Predicting Prices of the Apartments. ISPRS International Journal of Geo-Information, 7(5), 168.",
     "https://doi.org/10.3390/ijgi7050168"),
]


def money(value):
    return ("−" if value < 0 else "") + f"${abs(value):,.0f}"


def thousands(value):
    return ("−" if value < 0 else "") + f"${abs(value) / 1000:,.0f}k"


def read_results(folder):
    results = {path.stem: pd.read_csv(path) for path in folder.glob("*.csv")}
    for name, index in [("overall", "model"), ("intervals", "model"), ("selected", "model"),
                        ("cohort", "year"), ("monthly", "month")]:
        results[name] = results[name].set_index(index)
    for name in ["split", "filter_audit"]:
        results[name] = results[name].set_index(results[name].columns[0])["value"]
    return results


def chart_style(axis):
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#CCD9DC")
    axis.tick_params(length=0, labelcolor=INK)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#DFE8EA")


def make_figures(results, output):
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": MUTED})
    coverage = results["archive_coverage"].set_index("month").loc["2025-01":"2025-12"]
    fig, ax = plt.subplots(figsize=(9, 3.25), layout="constrained")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(months, coverage["Sales2025.zip"], color=GRAY, label="2025 archive")
    ax.bar(months, coverage["Sales2026.zip"], bottom=coverage["Sales2025.zip"],
           color=TEAL, label="2026 archive")
    ax.set_ylabel("Eligible transfers")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.14), frameon=False)
    ax.set_ylim(0, 365)
    chart_style(ax)
    fig.savefig(output / "coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.4), layout="constrained")
    values = results["overall"].loc[MODELS]
    x = np.arange(2)
    for i, (name, col) in enumerate(zip(MODELS, MODEL_COLORS)):
        bars = ax.bar(x + (i - 1) * .23, values.loc[name, ["mae", "rmse"]] / 1000,
                      .2, label=name, color=col)
        ax.bar_label(bars, fmt="$%.0fk", padding=4, color=INK)
    ax.set_xticks(x, ["Mean absolute error", "Root mean squared error"])
    ax.set_ylim(0, 535)
    ax.set_ylabel("Error ($ thousands)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.14), frameon=False)
    chart_style(ax)
    fig.savefig(output / "overall.png", dpi=180)
    plt.close(fig)

    price = results["price_band"].pivot(index="group", columns="model", values="bias")
    fig, ax = plt.subplots(figsize=(9, 3.45), layout="constrained")
    x = np.arange(3)
    for i, (name, col) in enumerate(zip(MODELS, MODEL_COLORS)):
        ax.bar(x + (i - 1) * .23, price.loc[["Low", "Middle", "High"], name] / 1000,
               .2, color=col, label=name)
    ax.axhline(0, color=INK, linewidth=.75)
    lower = thousands(results["split"]["lower_training_quartile"])
    upper = thousands(results["split"]["upper_training_quartile"])
    ax.set_xticks(x, [f"Up to {lower}", f"Over {lower} to {upper}", f"Above {upper}"])
    for label in ax.get_xticklabels():
        label.set_parse_math(False)
    ax.set_ylabel("Mean signed error ($ thousands)")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.16), frameon=False)
    chart_style(ax)
    fig.savefig(output / "bias.png", dpi=180)
    plt.close(fig)


def report(results, figures, output):
    overall, intervals, cohort = (results[name] for name in ["overall", "intervals", "cohort"])
    split, audit = results["split"], results["filter_audit"]
    group = lambda name: results[name].set_index(["group", "model"])
    forest_mae = overall.loc["Random forest", "mae"]
    gain = 100 * (1 - forest_mae / overall.loc["Median", "mae"])
    train, test = int(split["train_n"]), int(split["test_n"])
    styles = {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=31,
                                leading=34, textColor=colors.HexColor(INK), spaceAfter=16),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=23,
                             leading=27, textColor=colors.HexColor(INK), spaceAfter=13),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12,
                             leading=16, textColor=colors.HexColor(TEAL), spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.2,
                               leading=14.8, textColor=colors.HexColor(INK), spaceAfter=9, alignment=TA_LEFT),
        "small": ParagraphStyle("small", fontName="Helvetica", fontSize=9,
                                leading=12, textColor=colors.HexColor(MUTED), spaceAfter=8),
    }
    story = []

    def p(content, style="body"):
        story.append(Paragraph(content, styles[style]))

    def page(title):
        story.append(PageBreak())
        p(title, "h1")

    def table(headers, rows, widths):
        t = Table([headers] + rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(INK)),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor(PALE), colors.white]),
        ]))
        story.extend([t, Spacer(1, 10)])

    def picture(name, height):
        story.append(Image(str(figures / name), width=500, height=height, hAlign="LEFT"))

    p("CS 541A  /  FALL 2026  /  FINAL RESEARCH REPORT", "small")
    p("Predicting residential sale prices in Hudson County", "title")
    p("A temporal comparison using New Jersey public records", "h2")
    p(f"Can models trained on public records predict later Hudson County residential sale prices "
      f"more accurately than a municipality-level median? On {test:,} eligible calendar "
      f"2025 sales, the random forest reduced mean absolute error (MAE) by {gain:.1f}% to "
      f"{money(forest_mae)}. Ridge regression also improved MAE and RMSE. "
      "The forest's advantage was much larger when a pre-sale assessment record could be matched.")
    p("Study population and intended use", "h2")
    p("The comparison covers New Jersey SR-1A usable class 2 deed transfers in Hudson County. "
      "Class 2 and usable status are retrospective sample criteria. The findings describe that "
      "sample and the 2025 market. Students and researchers can use them to compare models and "
      "inspect where assessment-based estimates fail. This is a research prototype. Individual "
      "pricing, lending, tax assessment, and appraisal decisions are outside its scope.")
    p("How this compares with prior valuation work", "h2")
    p("Comparable-sales practice uses local transactions and adjusts for property differences. "
      "The municipality median captures locality with almost no data requirements, although it "
      "cannot adjust two properties within the same municipality. Zillow combines public records, "
      "MLS feeds, and user submissions in its Zestimate [5]. Those richer inputs and its "
      "proprietary model prevent a direct performance comparison.")
    p("Sirmans et al. [6] reviewed hedonic pricing studies and show why housing characteristics and "
      "location matter. Ridge provides a regularized version of a simple additive hedonic model "
      "here, with fewer property inputs than many such studies. Antipov and Pokryshevskaya [7] "
      "used a forest for apartment mass appraisal and CART to diagnose errors. Čeh et al. [8] "
      "reported lower errors from a forest than from multiple regression in Ljubljana apartments. "
      "Those findings motivate the model choices. Differences in local markets, property mix, "
      "features, and evaluation design limit any numerical comparison. This study evaluates "
      "later deeds and reports errors by municipality, price, match status and parcel overlap.")
    p("Project materials", "h2")
    p("The companion package includes Python source, source URLs and SHA-256 fingerprints, "
      "aggregate diagnostic tables, and the final presentation. The README documents how to rerun the study.")

    page("Data and feature timing")
    p("Calendar coverage and eligibility", "h2")
    p(f"The 2025 archive contains 1,452 eligible 2025 deed dates, through June. "
      "The 2026 archive adds 2,106, including the entire July–December portion. "
      "The 2027 year-to-date archive supplied zero eligible 2025 deeds in the version checked. "
      "Every 2025 month appears, with 256–334 sales per month. Coverage means every month "
      "is represented in the available state extracts. Late filings, corrections, and "
      "transfers outside the state's eligibility rules can still affect completeness.")
    picture("coverage.png", 172)
    p(f"The combined archives contain {int(audit['hudson_raw_rows']):,} Hudson rows. "
      f"The serial-based check removed {int(audit['serial_copies_removed'])} copies. "
      f"A second check on deed reference, date, parcel and price removed "
      f"{int(audit['additional_deed_copies_removed'])} Hudson copies across the source archives. "
      f"After deduplication, the deed-date window contains {int(audit['date_window_before_filters']):,} "
      f"Hudson rows and {int(audit['usable_class_2_in_window']):,} usable class 2 rows. "
      "The target is positive verified sale price, in nominal dollars. No eligible row "
      "failed that price check. Test prices ranged from $25,000 to $4.6 million. "
      "Both tails were retained because the files alone could not establish data errors.")
    p("Documented timing and match logic", "h2")
    p("Assessment values come from the same frozen 2023 MOD-IV archive for every fold. "
      "The Hudson member is dated June 5, 2023 and NJ's HTTP Last-Modified header is "
      "June 12, 2023 at 15:51:09 GMT. The July 1 start follows both dates. "
      "These archive dates document the apparent vintage of the sourced file. They do not "
      "independently establish the first public release date, correctness of individual fields, "
      "or whether a historical user had access on a given day. A municipality-only sensitivity "
      "check uses the field that can be independently known before the sale.")
    p("The join uses municipality, normalized block and lot (including suffixes), and the full "
      "qualifier. It preserves unmatched sales, removes conflicting assessment keys, "
      "and uses only primary class 2 assessment records. Multiple-parcel deeds use the primary "
      "parcel as their feature source. Parcel and deed identifiers never enter either model.")

    page("Models and temporal validation")
    p("Predictor audit", "h2")
    p("Municipality is known from location before sale. Year constructed, calculated acreage "
      "and net assessed value are recorded attributes from the frozen prior tax list. Net "
      "assessed value is a historical tax-roll value and may embed prior market information. "
      "None of the current transfer's price fields, sale ratio, fees, deed metadata, update "
      "fields, identities, or SR-1A living space/year built were used as predictors. "
      "The dated MOD-IV data available here do not supply the proposed living-space input.")
    keys = ["year_built_missing", "acres_missing", "net_value_missing"]
    table(["Deed period", "Sales", "Match rate", "Year missing", "Acres missing", "Value missing"],
          [[str(year) + (" H2" if year == 2023 else ""), f"{row['n']:,}",
            f"{row['matched'] / row['n']:.1%}",
            *[f"{row[field] / row['n']:.1%}" for field in keys]]
           for year, row in cohort.iterrows()], [105, 57, 78, 86, 82, 92])
    p("Missingness percentages use all eligible rows by deed year, before the training recording cutoff. "
      "Year-built values outside 1700–2023 and nonpositive acreage or assessment are treated as missing.", "small")
    p("Model definitions", "h2")
    p("The baseline predicts the training-period municipality median with the countywide "
      "training median as a fallback. Ridge is a linear model with L2 regularization on "
      "one-hot municipality indicators and standardized numeric inputs. The forest averages "
      "300 trees and uses one-hot municipality and unscaled numeric inputs. Both pipelines "
      "fit numeric median imputation and missingness indicators on training rows. Ridge scaling "
      "and municipality encoding are also fitted within each fold.")
    p("Split and selection", "h2")
    p(f"Expanding validation predicts 2024 H1 and 2024 H2. The folds use "
      "1,235 and 2,812 training rows and 1,644 and 1,503 validation rows, respectively. "
      "Training deeds had to be recorded before the relevant fold start. Validation labels "
      "had to be recorded before January 1, 2025. Mean of the two fold MAEs selected "
      "ridge alpha 10 from {1, 10, 100, 1000}, and minimum leaf size 5 from {5, 15, 40}. "
      f"The final fits use {train:,} training sales from July 2023 through 2024 after excluding "
      f"{int(split['train_deed_rows_excluded_by_recorded_cutoff'])} rows by the recording rule. "
      "The final test is January through December 2025 by deed date. Recording is a proxy "
      "for label availability. These extracts do not establish when the final usable flag, "
      "verified price, or transfer record became available online.")

    page("2025 holdout results")
    table(["Model", "Test n", "MAE", "RMSE", "Mean signed error"],
          [[name, f"{test:,}", *[money(overall.loc[name, metric]) for metric in ["mae", "rmse", "bias"]]]
           for name in MODELS], [145, 65, 94, 94, 102])
    picture("overall.png", 186)
    p("MAE averages the magnitude of errors and RMSE gives more weight to misses far from the sale "
      "price. Signed error is predicted minus observed. Negative values indicate average underprediction.")
    p("Paired uncertainty", "h2")
    p("Intervals use 2,000 paired percentile bootstrap draws of the test parcels, retaining "
      "their transfers together and holding fitted models fixed. The intervals quantify sample "
      "variation under a parcel-resampling assumption. They do not account for broader "
      "geographic dependence, market shifts, or training variation.")
    table(["Model minus median", "MAE difference, 95% interval", "RMSE difference, 95% interval"],
          [[name, f"{money(overall.loc[name, 'mae'] - overall.loc['Median', 'mae'])} "
                  f"({thousands(row['mae_low'])} to {thousands(row['mae_high'])})",
            f"{money(overall.loc[name, 'rmse'] - overall.loc['Median', 'rmse'])} "
            f"({thousands(row['rmse_low'])} to {thousands(row['rmse_high'])})"]
           for name, row in intervals.iterrows()], [130, 185, 185])
    p("Repeat-property overlap and feature sensitivity", "h2")
    overlap = group("overlap")
    matched = group("assessment")
    sensitiv = results["sensitivity"].set_index(["scenario", "model"])
    p(f"{int(overlap.loc[('Seen in training', 'Random forest'), 'n'])} test sales on "
      f"{int(split['test_distinct_seen_parcels'])} parcel keys appeared in final training. "
      f"The forest MAE on unseen parcel keys was "
      f"{money(overlap.loc[('Unseen parcel', 'Random forest'), 'mae'])}. "
      f"Removing overlapping parcels from training yielded forest MAE "
      f"{money(sensitiv.loc[('Training overlap removed', 'Random forest'), 'mae'])} "
      f"on the full test. This sensitivity deliberately uses the test keys to define the "
      "exclusion and keeps previously chosen settings. Parcel renumbering and split or "
      "combined lots can conceal overlap.")
    p(f"With municipality as the sole input, ridge and forest MAE were "
      f"{money(sensitiv.loc[('Municipality only', 'Ridge'), 'mae'])} and "
      f"{money(sensitiv.loc[('Municipality only', 'Random forest'), 'mae'])}. "
      "Neither beat median MAE under that specification.")

    page("Where the errors concentrate")
    municipalities = group("municipality")
    table(["Municipality", "Test n", "Median MAE", "Ridge MAE", "Forest MAE"],
          [[name, f"{int(municipalities.loc[(name, 'Median'), 'n']):,}",
            *[money(municipalities.loc[(name, model), "mae"]) if
              pd.notna(municipalities.loc[(name, model), "mae"]) else "—" for model in MODELS]]
           for name in sorted(results["municipality"]["group"].unique())], [155, 60, 95, 95, 95])
    p("East Newark had fewer than 30 test sales. Its error metrics are suppressed. The forest had "
      "lower MAE than the median in all 11 municipalities with reported results. Jersey City "
      "contributed about 45% of the test sample, so pooled metrics are influenced by its results.", "small")
    picture("bias.png", 174)
    p("Price groups use training quartiles of $450,000 and $880,000 and are assigned using "
      "observed test prices. Forest MAE was "
      f"{money(group('price_band').loc[('Low', 'Random forest'), 'mae'])} for 762 low-price sales, "
      f"{money(group('price_band').loc[('Middle', 'Random forest'), 'mae'])} for 1,787 middle sales, and "
      f"{money(group('price_band').loc[('High', 'Random forest'), 'mae'])} for 1,009 high-price sales. "
      "The forest overpredicted at the low end and underpredicted at the high end. "
      "Conditioning on the realized price can produce this pattern through regression toward "
      "the mean, so these bins describe realized errors and are not available at prediction time.")

    page("Interpretation and references")
    p(f"Matched sales had forest MAE of {money(matched.loc[('Matched', 'Random forest'), 'mae'])} "
      f"versus {money(matched.loc[('Unmatched', 'Random forest'), 'mae'])} for unmatched sales. "
      f"For the {int(group('parcel_scope').loc[('Additional lot', 'Random forest'), 'n'])} "
      f"deeds reporting additional lots, forest MAE reached "
      f"{money(group('parcel_scope').loc[('Additional lot', 'Random forest'), 'mae'])}. "
      "Matching and parcel structure are associated with market composition and missingness. "
      "These subgroup comparisons do not isolate causal effects.")
    p("The results support an improvement over a simple municipal benchmark for these data. "
      "The largest gains came with linked assessments. The assessment vintage misses later "
      "construction and renovations, and the models lack condition, floor area, unit amenities, "
      "nearby sales features, and detailed location. The eligible-sales filter also limits "
      "generalization to other transfer types. Evidence from another county, year, or an actual "
      "pre-sale release archive would answer a different and more demanding question.")
    p("Sources", "h2")
    for i, (label, url) in enumerate(REFERENCES, 1):
        p(f'{i}. {escape(label)}<br/><link href="{url}" color="{TEAL}">{escape(url)}</link>', "small")
    p("Assistance disclosure", "h2")
    p("OpenAI Codex assisted with implementation, analysis, and preparation of this report "
      "and the presentation from the supplied proposal. Course AI-use requirements were "
      "unavailable during preparation.", "small")

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D7E4E3"))
        canvas.line(56, 42, 556, 42)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(56, 27, "CS 541A  /  HUDSON COUNTY SALE PRICE STUDY")
        canvas.drawRightString(556, 27, str(doc.page))

    doc = SimpleDocTemplate(str(output), title="Predicting residential sale prices in Hudson County",
                            leftMargin=56, rightMargin=56, topMargin=47, bottomMargin=54,
                            pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def presentation(results, figures, output):
    deck = Presentation()
    deck.slide_width = Inches(13.3333)
    deck.slide_height = Inches(7.5)
    overall, interval = results["overall"], results["intervals"]
    split = results["split"]
    cohort = results["cohort"]
    forest, baseline = overall.loc["Random forest"], overall.loc["Median"]
    gain = 1 - forest["mae"] / baseline["mae"]

    def rgb(color):
        return RGBColor.from_string(color.lstrip("#"))

    def text(slide, content, x, y, w, h, size, color=INK, bold=False):
        frame = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = True
        for i, line in enumerate(content.split("\n")):
            paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            paragraph.text = line
            paragraph.font.name = "DejaVu Sans"
            paragraph.font.size = Pt(size)
            paragraph.font.color.rgb = rgb(color)
            paragraph.font.bold = bold
            paragraph.space_after = Pt(7)

    def rect(slide, x, y, w, h, color):
        block = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        block.fill.solid()
        block.fill.fore_color.rgb = rgb(color)
        block.line.fill.background()

    def slide(heading=None):
        item = deck.slides.add_slide(deck.slide_layouts[6])
        item.background.fill.solid()
        item.background.fill.fore_color.rgb = rgb(WHITE)
        rect(item, 0, 0, 13.34, .12, TEAL)
        if heading:
            text(item, heading, .7, .55, 12, .65, 29, INK, True)
        rect(item, .7, 7.03, 11.93, .013, "#D7E4E3")
        text(item, "CS 541A  /  HUDSON COUNTY SALE PRICE STUDY", .7, 7.15, 9.0, .23, 9, MUTED, True)
        text(item, str(len(deck.slides)), 12.2, 7.15, .4, .25, 9, MUTED)
        return item

    def items(item, pairs, start=1.6, spacing=1.55):
        for i, (heading, body) in enumerate(pairs):
            yy = start + i * spacing
            text(item, heading, .75, yy, 11.8, .38, 17, TEAL, True)
            text(item, body, .75, yy + .43, 11.85, spacing - .46, 17, INK)

    def picture(item, name, y=1.33, w=10.9):
        item.shapes.add_picture(str(figures / name), Inches((13.333 - w) / 2), Inches(y), width=Inches(w))

    def note(item, body):
        rect(item, .7, 6.04, 11.93, .79, PALE)
        text(item, body, .94, 6.19, 11.45, .55, 15, TEAL, True)

    item = slide()
    text(item, "FINAL RESEARCH PRESENTATION  /  FALL 2026", .74, .88, 11.6, .4, 15, TEAL, True)
    text(item, "Predicting residential sale prices\nin Hudson County", .72, 1.48, 11.9, 2.05, 43, INK, True)
    text(item, "Can public records improve on a municipality-level median?", .75, 3.76, 11.8, .8, 24, MUTED)
    rect(item, .75, 5.0, 11.8, 1.26, PALE)
    text(item, f"{gain:.1%} lower 2025 MAE with a random forest", 1.03, 5.35, 11.15, .5, 25, TEAL, True)

    item = slide("Why test these models?")
    items(item, [
        ("A local median is an accessible benchmark",
         "It uses earlier municipality sales and cannot distinguish two homes within that municipality."),
        ("Hedonic regression and random forests answer different modeling questions",
         "Ridge regularizes an additive model. Forests capture nonlinear relationships and interactions. Prior apartment studies motivate the comparison [6–8]."),
        ("Commercial valuations use different information",
         "Zillow combines MLS, public records and user submissions. Its estimates are outside the comparison [5]."),
    ])

    item = slide("Two archives cover the 2025 deed year")
    picture(item, "coverage.png")
    note(item, "1,452 sales in the 2025 archive + 2,106 in the 2026 archive = 3,558.")
    text(item, "Calendar coverage is verified in the extracts. Late or missing transfers can still affect completeness.",
         .95, 5.51, 11.5, .37, 13, MUTED)

    item = slide("Use property fields from an earlier snapshot")
    items(item, [
        ("Recorded before the study window",
         "The frozen 2023 MOD-IV Hudson file is dated June 5, 2023. The state server's file timestamp is June 12. First historical public release was not independently verified."),
        ("Three assessment inputs and municipality",
         "Year built, calculated acreage and net assessed value came from the assessment file. SR-1A sale-derived fields and living area were excluded."),
        ("Link on the parcel key, then drop it from modeling",
         f"Match by district, block, lot and qualifier. In 2025, {cohort.loc[2025, 'matched'] / cohort.loc[2025, 'n']:.1%} matched. "
         "Unmatched sales stay in the evaluation."),
    ])

    item = slide("Validate in time order, then test on 2025")
    items(item, [
        ("July–December 2023 starts the development period",
         "Use the first half and second half of 2024 as separate expanding-window validation periods."),
        ("Choose settings using validation MAE",
         "The selected ridge alpha is 10 and the forest minimum leaf size is 5. Imputation and encoding are fitted inside the training pipeline."),
        (f"{int(split['train_n']):,} final training sales and {int(split['test_n']):,} later test sales",
         "Training transfers were recorded before the cutoff. The test is defined by deed date and scored in nominal dollars."),
    ])

    item = slide("The forest had the lowest 2025 error")
    picture(item, "overall.png")
    ci = interval.loc["Random forest"]
    note(item, f"Forest versus median MAE difference  {thousands(forest['mae'] - baseline['mae'])}  "
               f"(95% parcel bootstrap {thousands(ci['mae_low'])} to {thousands(ci['mae_high'])}).")

    item = slide("Error changes with price and location")
    picture(item, "bias.png", w=10.65)
    text(item, "Signed error is prediction minus sale price. Price bands use training quartiles and observed test prices.",
         .95, 5.55, 11.5, .38, 13, MUTED)
    note(item, "Forest MAE beat the median in all 11 reported municipalities. East Newark had 7 sales.")

    item = slide("The assessment match matters")
    assessment = results["assessment"].set_index(["group", "model"])
    overlap = results["overlap"].set_index(["group", "model"])
    sensitive = results["sensitivity"].set_index(["scenario", "model"])
    items(item, [
        ("Matched versus unmatched assessment records",
         f"Forest MAE was {thousands(assessment.loc[('Matched', 'Random forest'), 'mae'])} "
         f"for matched sales and {thousands(assessment.loc[('Unmatched', 'Random forest'), 'mae'])} for unmatched sales."),
        ("Repeat parcel overlap was small",
         f"{int(overlap.loc[('Seen in training', 'Random forest'), 'n'])} test transfers shared a parcel key with training. "
         f"Refitting without those training parcels gave {thousands(sensitive.loc[('Training overlap removed', 'Random forest'), 'mae'])} test MAE."),
        ("Municipality alone did not improve on median MAE",
         f"Without assessment fields the forest reached {thousands(sensitive.loc[('Municipality only', 'Random forest'), 'mae'])}, "
         f"compared with {thousands(baseline['mae'])} for the local median."),
    ])

    item = slide("What the comparison supports")
    items(item, [
        ("Dated assessments improved the retrospective comparison",
         "The forest reduced both MAE and RMSE. Missing records and high-price underprediction limit precision for individual properties."),
        ("The sample is market- and time-specific",
         "Class 2 usable transactions, one county, one test year and a frozen assessment vintage limit generalization. Exact label release timing remains unknown."),
        ("Research use",
         "The scripts, audit tables and report let students and researchers reproduce the comparison and inspect errors. The study does not support individual valuation decisions."),
    ])

    item = slide("Sources and reproducibility")
    for i, (label, url) in enumerate(REFERENCES, 1):
        short = [
            "New Jersey Division of Taxation · Statistical Information and archives",
            "New Jersey Division of Taxation · SR-1A layout",
            "New Jersey Division of Taxation · MOD-IV layout",
            "Rutgers University · N.J. MOD IV Historical Database",
            "Zillow · What is a Zestimate?",
            "Sirmans et al. (2005) · The Composition of Hedonic Pricing Models",
            "Antipov and Pokryshevskaya (2012) · Mass appraisal of residential apartments",
            "Čeh et al. (2018) · Random forest versus multiple regression in apartment sales",
        ][i - 1]
        frame = item.shapes.add_textbox(Inches(.84), Inches(1.51 + (i - 1) * .49),
                                       Inches(11.5), Inches(.4)).text_frame
        frame.margin_left = frame.margin_top = 0
        run = frame.paragraphs[0].add_run()
        run.text = f"[{i}]  {short}"
        run.font.name = "DejaVu Sans"
        run.font.size = Pt(15)
        run.font.color.rgb = rgb(TEAL)
        run.hyperlink.address = url
    note(item, "OpenAI Codex assisted with the implementation, analysis and presentation.")
    deck.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    results = read_results(args.results)
    make_figures(results, figures)
    report(results, figures, args.output / "report.pdf")
    presentation(results, figures, args.output / "presentation.pptx")


if __name__ == "__main__":
    main()
