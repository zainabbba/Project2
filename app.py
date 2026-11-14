"""Afham Reviews — Arabic sentiment and topic analysis for CSV uploads."""
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from Helper.helper import clean_review, load_model, normalize_sentiment, sentiment_counts, top_3_complaints

ROOT = Path(__file__).resolve().parent
SENTIMENT = ROOT / "Models" / "sentiment_pipeline.joblib"
TOPIC = ROOT / "Models" / "topic_pipeline.joblib"
SAMPLE_PATH = ROOT / "Data" / "sample_reviews.csv"

st.set_page_config(page_title="افهم تقييماتك", page_icon="☕", layout="wide")
st.markdown("""
<style>
.stApp {direction:rtl;text-align:right;background:#f6f4ec;color:#183c32}
[data-testid="stMainBlockContainer"] {max-width:1120px;padding-top:3rem}
.hero {background:linear-gradient(125deg,#103a30,#246552);color:#fff9e9;
padding:clamp(22px,5vw,54px);border-radius:26px;margin-bottom:24px}
.hero h1 {color:inherit;font-size:clamp(32px,6vw,60px);margin:0;padding:0}
.hero p {margin:.6rem 0 0;color:#e0eddf}
[data-testid="stMetric"] {background:#fffdf8;border:1px solid #dddcd0;
padding:14px;border-radius:16px}
[data-testid="stFileUploaderDropzone"] {border:2px dashed #95aa9c;border-radius:14px}
</style>
<header class="hero"><h1>☕ افهم تقييماتك</h1>
<p>تصنيف مشاعر العملاء ومواضيع تقييماتهم وتحليل أكثر فئات الشكاوى تكرارًا.</p></header>
""", unsafe_allow_html=True)


@st.cache_resource
def cached_model(path: str, mtime: int):
    """Cache only trusted models; invalidate cache when their files change."""
    return load_model(path)


def read_reviews(content: bytes) -> pd.DataFrame:
    # Try common CSV delimiters, while requiring a unique Review column.
    for encoding in ("utf-8-sig", "utf-8"):
        for sep in (",", ";", "\t"):
            try:
                data = pd.read_csv(BytesIO(content), sep=sep, encoding=encoding,
                                   dtype=str, keep_default_na=False)
            except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            data.columns = data.columns.str.strip()
            if data.columns.is_unique and "Review" in data.columns:
                return data
    raise ValueError("ارفعي ملف CSV بترميز UTF-8 ويحتوي على عمود Review.")


with st.container(border=True):
    st.subheader("رفع ملف التقييمات")
    st.caption("يكفي عمود Review فقط؛ سيتوقع التطبيق المشاعر والفئة تلقائيًا.")
    if SAMPLE_PATH.is_file():
        st.download_button(
            "⬇️ تحميل ملف تجريبي CSV",
            data=SAMPLE_PATH.read_bytes(),
            file_name="sample_reviews.csv",
            mime="text/csv",
        )
    uploaded = st.file_uploader("ملف التقييمات", type=["csv"], label_visibility="collapsed")

if uploaded is None:
    st.info("حمّلي ملف التجربة، ثم ارفعيه هنا لتحليل التعليقات.")
    st.stop()

try:
    data = read_reviews(uploaded.getvalue())
except ValueError as exc:
    st.error(str(exc))
    st.stop()

reviews = data["Review"].map(clean_review)
valid = reviews.ne("")
if not valid.any():
    st.warning("لا توجد تعليقات صالحة في عمود Review.")
    st.stop()

missing = [str(p.relative_to(ROOT)) for p in (SENTIMENT, TOPIC) if not p.is_file()]
if missing:
    st.error("ملفات المودل غير موجودة: " + "، ".join(missing))
    st.info("شغّلي نوتبوك التدريب حتى خطوة حفظ المودلز، ثم ضعي الملفين في مجلد Models.")
    st.stop()

with st.spinner("جارٍ تصنيف المشاعر والمواضيع..."):
    sentiment_model = cached_model(str(SENTIMENT), SENTIMENT.stat().st_mtime_ns)
    topic_model = cached_model(str(TOPIC), TOPIC.stat().st_mtime_ns)
    results = data.loc[valid].copy()
    cleaned = reviews.loc[valid]
    results["Predicted_Sentiment"] = [normalize_sentiment(x) for x in sentiment_model.predict(cleaned)]
    results["Predicted_Attribute"] = topic_model.predict(cleaned)

st.subheader("رضا العملاء حسب التوقعات")
st.caption(f"من أصل {len(results):,} تعليق محلّل. النسب تقديرية وتعتمد على أداء المودل.")
counts = sentiment_counts(results)
labels = {"إيجابي": "نسبة الراضين", "سلبي": "نسبة غير الراضين", "محايد": "نسبة المحايدين"}
for card, (label, count) in zip(st.columns(3), counts.items()):
    with card:
        st.metric(labels[label], f"{count / len(results):.1%}")
        st.caption(f"{count:,} تعليق — {label}")

with st.container(border=True):
    st.subheader("أكثر ٣ فئات للشكاوى")
    top, negative_count = top_3_complaints(results)
    if negative_count == 0:
        st.info("لا توجد تعليقات توقعها المودل سلبية.")
    elif top.empty:
        st.info("لم يستطع المودل تحديد مواضيع للتعليقات السلبية.")
    else:
        st.caption("الفئات مرتبة حسب عدد التعليقات المصنفة سلبية. يأخذ كل تعليق فئة واحدة حاليًا.")
        for rank, (topic, count) in enumerate(top.items(), start=1):
            fraction = count / negative_count
            st.progress(float(fraction), text=f"{rank}. {topic} — {count} شكوى ({fraction:.1%} من السلبي)")
        st.bar_chart(top.rename("Negative Reviews"))
        st.info(f"توصية أولية: ابدئي بمراجعة تعليقات «{top.index[0]}» والتحقق يدويًا من الأسباب المتكررة.")

st.subheader("نتائج تصنيف التعليقات")
st.dataframe(results, hide_index=True, use_container_width=True)
st.download_button(
    "⬇️ تحميل النتائج CSV",
    data=results.to_csv(index=False).encode("utf-8-sig"),
    file_name="reviews_analyzed.csv",
    mime="text/csv",
)
with st.expander("كيف احتُسبت النتائج؟"):
    st.write("يتوقع مودل Result المشاعر ويتوقع مودل Attribute الفئة. بعدها نحسب أكثر ثلاث فئات ضمن التعليقات المتوقعة سلبية. كل تعليق يُسند إلى موضوع واحد، لذا لا تشمل النتائج كل المواضيع التي قد يذكرها تعليق متعدد الجوانب. النتائج تقديرية ويجب التحقق من التعليقات المهمة يدويًا.")
