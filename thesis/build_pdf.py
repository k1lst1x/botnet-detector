"""Сборка дипломной работы из thesis/*.md в DOCX и PDF по правилам МУИТ R-15 (ред. 3).

Требует: python-docx, pypdf, LibreOffice (soffice).
Запуск: python thesis/build_pdf.py  → thesis/build/diploma.{docx,pdf}

Оформление (R-15, п. 7 и замечания нормоконтроля):
  A4; поля: левое 30, правое 10, верхнее 20, нижнее 25 мм; Times New Roman 14; одинарный интервал;
  номер страницы внизу по центру, на титульном листе не ставится, обложка не нумеруется;
  структурные заголовки — прописными по центру; разделы — с абзацного отступа прописными;
  подразделы — с абзацного отступа; две пустые строки перед подразделом, одна после;
  каждый раздел с новой страницы; «Таблица 1.1 – …» над таблицей с абзацного отступа;
  «Рисунок 1.1 – …» под рисунком по центру; литература по IEEE в порядке упоминания;
  аннотации в порядке: казахский, русский, английский.

Разметка Markdown:
  # ВВЕДЕНИЕ                — структурный элемент;   # 1 НАЗВАНИЕ — раздел;
  ## 1.1 …, ### 1.1.1 …     — подразделы;            # ПРИЛОЖЕНИЕ А | Заголовок — приложение;
  Таблица 1.1 – …           — подпись к следующей таблице;
  ![Рисунок 1.1 – …](путь)  — рисунок;               $$ … $$ — формула отдельной строкой;
  [@key; @key2]             — ссылки на источники из references.md.
"""
import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"
FONT = "Times New Roman"
LINE = Pt(16)  # одна пустая строка при кегле 14 и одинарном интервале
INDENT = Cm(1.25)
TEXT_WIDTH_CM = 21 - 3 - 1

META = {
    "ministry": "МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РЕСПУБЛИКИ КАЗАХСТАН",
    "university": "МЕЖДУНАРОДНЫЙ УНИВЕРСИТЕТ ИНФОРМАЦИОННЫХ ТЕХНОЛОГИЙ",
    "faculty": "ФАКУЛЬТЕТ КОМПЬЮТЕРНЫХ ТЕХНОЛОГИЙ И КИБЕРБЕЗОПАСНОСТИ",
    "department": "КАФЕДРА [НАЗВАНИЕ КАФЕДРЫ]",
    "head": "[Фамилия И.О.]",
    "student": "[Фамилия И.О.]",
    "advisor": "[Фамилия И.О.]",
    "advisor_title": "[учёная степень, должность]",
    "title": "Разработка системы обнаружения скоординированных бот-сетей в социальных сетях "
             "на основе анализа поведенческих и сетевых признаков",
    "program": "[шифр] [название образовательной программы]",
    "kind": "ДИПЛОМНАЯ РАБОТА",
    "city_year": "Алматы 2027",
}

INITIAL_DATA = ("Открытый набор данных TwiBot-22 (профили и публикации аккаунтов Twitter с разметкой); "
                "публичные данные аккаунтов Threads; программные средства Python, Django, FastAPI, Celery, XGBoost.")
TASKS = [
    "Анализ понятий социального бота и бот-сети, существующих методов их обнаружения, наборов данных и метрик.",
    "Выбор поведенческих и сетевых признаков, наблюдаемых в Twitter и Threads.",
    "Подготовка обучающей выборки на основе TwiBot-22.",
    "Обучение и сравнение моделей классификации аккаунтов, оценка вклада признаков.",
    "Построение графа совместной активности и выделение скоординированных групп.",
    "Разработка программной системы с интерфейсом ручной разметки.",
    "Проверка переноса модели на размеченную выборку аккаунтов Threads.",
]
SCHEDULE = [
    ("Составление календарного плана выполнения дипломной работы", "30 ноября"),
    ("Сбор, изучение, обработка и анализ материалов", "ноябрь – декабрь"),
    ("Подготовка и сдача руководителю введения, разделов 1–3, заключения", "январь – февраль"),
    ("Доработка дипломной работы с учётом замечаний руководителя", "март – апрель"),
    ("Сдача законченной дипломной работы руководителю", "15 апреля"),
    ("Предзащита", "апрель"),
    ("Нормоконтроль", "30 апреля – 15 мая"),
    ("Проверка на заимствования", "3 мая – 24 мая"),
    ("Сдача работы на рецензию", "3 мая – 24 мая"),
    ("Защита дипломной работы", "24 мая – 19 июня"),
]


# ---------- низкоуровневое форматирование ----------

def set_font(run, size=14, bold=None, italic=None, mono=False):
    name = "Courier New" if mono else FONT
    run.font.name = name
    run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), name)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_inline(par, text, size=14, bold=False, italic=False):
    """***bold italic***, **bold**, *italic*, `code`."""
    for tok in re.split(r"(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("***"):
            set_font(par.add_run(tok[3:-3]), size, bold=True, italic=True)
        elif tok.startswith("**"):
            set_font(par.add_run(tok[2:-2]), size, bold=True, italic=italic)
        elif tok.startswith("`"):
            set_font(par.add_run(tok[1:-1]), size - 1, bold=bold, mono=True)
        elif tok.startswith("*") and len(tok) > 2:
            set_font(par.add_run(tok[1:-1]), size, bold=bold, italic=True)
        else:
            set_font(par.add_run(tok), size, bold=bold, italic=italic)


def fmt(par, indent=INDENT, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=0, left=None,
        keep=False, new_page=False):
    pf = par.paragraph_format
    pf.first_line_indent = indent
    pf.alignment = align
    pf.line_spacing = 1.0
    pf.space_before = before if hasattr(before, "pt") else Pt(before)
    pf.space_after = after if hasattr(after, "pt") else Pt(after)
    if left is not None:
        pf.left_indent = left
    pf.keep_with_next = keep
    pf.page_break_before = new_page
    pf.widow_control = True


def para(doc, text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=INDENT, size=14, bold=False, italic=False,
         before=0, after=0, keep=False, new_page=False, left=None):
    p = doc.add_paragraph()
    fmt(p, indent=indent, align=align, before=before, after=after, keep=keep, new_page=new_page, left=left)
    if text:
        add_inline(p, text, size, bold, italic)
    return p


def center(doc, text="", **kw):
    return para(doc, text, align=WD_ALIGN_PARAGRAPH.CENTER, indent=Cm(0), **kw)


def no_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tbl_pr.append(borders)


def make_table(doc, rows, widths_cm, size=12, header=True, borders=True, aligns=None):
    t = doc.add_table(rows=len(rows), cols=len(widths_cm))
    if borders:
        t.style = "Table Grid"
    else:
        no_borders(t)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    if header and borders:
        tr_pr = t.rows[0]._tr.get_or_add_trPr()
        rep = OxmlElement("w:tblHeader")
        rep.set(qn("w:val"), "true")
        tr_pr.append(rep)
    for c, col in enumerate(t.columns):
        col.width = Cm(widths_cm[c])
    for r, row in enumerate(rows):
        for c in range(len(widths_cm)):
            cell = t.cell(r, c)
            cell.width = Cm(widths_cm[c])
            p = cell.paragraphs[0]
            align = (aligns[c] if aligns else WD_ALIGN_PARAGRAPH.LEFT)
            if header and r == 0 and borders:
                align = WD_ALIGN_PARAGRAPH.CENTER
            fmt(p, indent=Cm(0), align=align, keep=len(rows) <= 10 and r < len(rows) - 1)
            add_inline(p, row[c] if c < len(row) else "", size, bold=False)
    return t


def page_number_footer(section):
    section.footer.is_linked_to_previous = False
    par = section.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run()
    set_font(run, 14)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._element.append(el)


def setup_section(sec):
    sec.page_height, sec.page_width = Cm(29.7), Cm(21)
    sec.left_margin, sec.right_margin = Cm(3), Cm(1)
    sec.top_margin, sec.bottom_margin = Cm(2), Cm(2.5)


# ---------- разбор Markdown ----------

def parse(md):
    blocks, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,3}) (.*)", line)
        if m:
            blocks.append((f"h{len(m.group(1))}", m.group(2).strip()))
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", rows))
            continue
        if re.match(r"^Таблица \S+ – ", line):
            blocks.append(("tcap", line.strip()))
            i += 1
            continue
        fm = re.match(r"^!\[(.*)\]\((.*)\)\s*$", line)
        if fm:
            blocks.append(("fig", (fm.group(1), fm.group(2))))
            i += 1
            continue
        if line.startswith("$$"):
            blocks.append(("formula", line.strip().strip("$").strip()))
            i += 1
            continue
        lm = re.match(r"^(- |\d+\. )(.*)", line)
        if lm:
            kind = "ul" if lm.group(1) == "- " else "ol"
            items = []
            while i < len(lines) and lines[i].strip():
                im = re.match(r"^(- |\d+\. )(.*)", lines[i])
                if im:
                    items.append([im.group(1).strip(), im.group(2)])
                else:
                    items[-1][1] += " " + lines[i].strip()
                i += 1
            blocks.append((kind, items))
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |\d+\. |\$\$|!\[)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(buf)))
    return blocks


def load_references():
    refs = {}
    for line in (HERE / "references.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([\w-]+): (.+)$", line)
        if m:
            refs[m.group(1)] = m.group(2)
    return refs


class Citations:
    def __init__(self, refs):
        self.refs, self.order = refs, []

    def number(self, key):
        if key not in self.refs:
            raise SystemExit(f"Нет источника: {key}")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def render(self, text):
        def repl(m):
            nums = sorted({self.number(k.strip().lstrip("@")) for k in m.group(1).split(";")})
            parts, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                    j += 1
                parts.append(f"[{nums[i]}]–[{nums[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in nums[i:j + 1]))
                i = j + 1
            return ", ".join(parts)
        return re.sub(r"\[(@[^\]]+)\]", repl, text)


def ru_plural(n, one, few, many):
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


# ---------- документ ----------

class Builder:
    def __init__(self, pages=None, stats=None):
        self.pages = pages or {}
        self.stats = stats or {"pages": 0, "figures": 0, "tables": 0, "refs": 0, "apps": 0}
        self.doc = Document()
        self.toc = []  # (номер, текст, ключ поиска, уровень)
        self.fig_count = self.table_count = 0
        self.cit = Citations(load_references())

    # --- первые страницы ---
    def cover(self):
        d = self.doc
        for line in (META["ministry"], META["university"], META["faculty"]):
            center(d, line)
        para(d, META["student"], indent=Cm(0), align=WD_ALIGN_PARAGRAPH.LEFT, before=110)
        center(d, META["title"], bold=True, before=70)
        center(d, META["kind"], before=40)
        center(d, f"Образовательная программа: {META['program']}", bold=True, before=40)
        center(d, META["city_year"], before=150)

    def title_page(self):
        d = self.doc
        center(d, META["ministry"], new_page=True)
        for line in (META["university"], META["faculty"], META["department"]):
            center(d, line)
        left = Cm(10)
        para(d, "Допущен к защите", indent=Cm(0), left=left, before=32, align=WD_ALIGN_PARAGRAPH.LEFT)
        para(d, "Заведующий кафедрой", indent=Cm(0), left=left, align=WD_ALIGN_PARAGRAPH.LEFT)
        para(d, f"____________ {META['head']}", indent=Cm(0), left=left, align=WD_ALIGN_PARAGRAPH.LEFT)
        para(d, "«____» ____________ 20____ г.", indent=Cm(0), left=left, align=WD_ALIGN_PARAGRAPH.LEFT)
        center(d, META["kind"], bold=True, before=60)
        center(d, META["title"], bold=True, before=16)
        center(d, f"Образовательная программа: {META['program']}", bold=True, before=16)
        rows = [
            ["Выполнил:", "", ""],
            [META["student"], "____________", "«____» ____________ 20____ г."],
            ["", "(подпись)", ""],
            ["Научный руководитель:", "", ""],
            [f"{META['advisor']}, {META['advisor_title']}", "____________", "«____» ____________ 20____ г."],
            ["", "(подпись)", ""],
        ]
        para(d, "", before=30)
        t = make_table(d, rows, [6.0, 3.5, 7.5], size=14, header=False, borders=False,
                       aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT])
        for r in (2, 5):
            for p in t.cell(r, 1).paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
        center(d, META["city_year"], before=50)

    def assignment(self):
        d = self.doc
        center(d, "Международный университет информационных технологий", new_page=True)
        center(d, "Факультет компьютерных технологий и кибербезопасности")
        center(d, "Кафедра [название кафедры]")
        para(d, f"Образовательная программа: {META['program']}", indent=Cm(0), before=16)
        center(d, "ЗАДАНИЕ НА ДИПЛОМНУЮ РАБОТУ", bold=True, before=16, after=16)
        para(d, f"Обучающийся: **{META['student']}**", indent=Cm(0))
        para(d, f"Тема дипломной работы: **{META['title']}**", indent=Cm(0), before=8, align=WD_ALIGN_PARAGRAPH.LEFT)
        para(d, "Утверждена приказом МУИТ № ______ от «____» ____________ 20____ г.", indent=Cm(0), before=8)
        para(d, "Срок сдачи законченной работы: «____» ____________ 20____ г.", indent=Cm(0), before=8)
        para(d, f"Исходные данные к работе: {INITIAL_DATA}", indent=Cm(0), before=8)
        para(d, "Перечень вопросов, подлежащих разработке:", indent=Cm(0), before=8)
        for n, task in enumerate(TASKS, 1):
            para(d, f"{n}. {task}", indent=Cm(0))
        para(d, "Консультации по работе с указанием относящихся к ним разделов:", indent=Cm(0), before=8, after=6, keep=True)
        make_table(d, [["Консультант", "Ф.И.О.", "Задание выдал (подпись, дата)", "Задание принял (подпись, дата)"],
                       ["Нормоконтроль", "", "", ""]], [3.5, 4.5, 4.5, 4.5], size=12)
        para(d, "Дата выдачи задания «____» ____________ 20____ г.", indent=Cm(0), before=16)
        para(d, "Научный руководитель ________________ (подпись)", indent=Cm(0), before=8)
        para(d, "Задание принял к исполнению ________________ (подпись)", indent=Cm(0), before=8)

    def schedule(self):
        d = self.doc
        center(d, "КАЛЕНДАРНЫЙ ПЛАН ВЫПОЛНЕНИЯ ДИПЛОМНОЙ РАБОТЫ", new_page=True, after=16)
        para(d, f"Обучающийся: **{META['student']}**", indent=Cm(0))
        para(d, f"Тема: **{META['title']}**", indent=Cm(0), after=16, align=WD_ALIGN_PARAGRAPH.LEFT)
        rows = [["№", "Этап работы", "Срок выполнения"]] + [[f"{i}", s, t] for i, (s, t) in enumerate(SCHEDULE, 1)]
        make_table(d, rows, [1.0, 11.0, 5.0], size=12)
        para(d, f"Обучающийся ________________ {META['student']}", indent=Cm(0), before=32)
        para(d, f"Научный руководитель ________________ {META['advisor']}", indent=Cm(0), before=16)
        para(d, "«____» ____________ 20____ г.", indent=Cm(0), before=16)

    def annotations(self):
        s = self.stats
        def join(items, conj):
            if not items:
                return "…"
            return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {conj} " + items[-1]

        counts = [("pages", "беттен", ("страницу", "страницы", "страниц"), "page"),
                  ("figures", "суреттен", ("рисунок", "рисунка", "рисунков"), "figure"),
                  ("tables", "кестеден", ("таблицу", "таблицы", "таблиц"), "table"),
                  ("refs", "әдебиет көзінен", ("источник", "источника", "источников"), "reference"),
                  ("apps", "қосымшадан", ("приложение", "приложения", "приложений"), "appendix")]
        present = [c for c in counts if s[c[0]]]
        kk = join([f"{s[k]} {w}" for k, w, _, _ in present], "және")
        ru = ", ".join(f"{s[k]} {ru_plural(s[k], *forms)}" for k, _, forms, _ in present)
        en = join([f"{s[k]} {w}{'' if s[k] == 1 else ('es' if w == 'appendix' else 's')}" for k, _, _, w in present], "and")
        texts = {
            "АҢДАТПА": f"Дипломдық жұмыс {kk} тұрады.",
            "АННОТАЦИЯ": f"Дипломная работа содержит {ru}.",
            "ANNOTATION": f"The diploma work contains {en}.",
        }
        heading = None
        for kind, payload in parse((HERE / "front" / "annotations.md").read_text(encoding="utf-8")):
            if kind == "h1":
                heading = payload
                center(self.doc, payload, new_page=True, after=LINE)
            elif kind == "p":
                para(self.doc, payload.replace("{stats}", texts[heading]))

    def contents(self):
        d = self.doc
        center(d, "СОДЕРЖАНИЕ", new_page=True, after=LINE)
        rows = [[num, text, str(self.pages.get(key, "00"))] for num, text, key, _ in self.toc]
        t = make_table(d, rows, [1.6, 14.2, 1.2], size=14, header=False, borders=False,
                       aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT])
        return t

    # --- основная часть ---
    def heading(self, kind, text, prev_kind):
        d = self.doc
        if kind == "h1":
            m = re.match(r"^ПРИЛОЖЕНИЕ (\S+) \| (.+)$", text)
            if m:
                center(d, f"ПРИЛОЖЕНИЕ {m.group(1)}", new_page=True)
                center(d, m.group(2), bold=True, after=LINE)
                self.stats_apps += 1
                self.toc.append(("", f"Приложение {m.group(1)}. {m.group(2)}", f"ПРИЛОЖЕНИЕ {m.group(1)}", 0))
                return
            num = re.match(r"^(\d+) (.+)$", text)
            if num:
                para(d, text.upper(), bold=True, new_page=True, after=LINE, keep=True, align=WD_ALIGN_PARAGRAPH.LEFT)
                self.toc.append((num.group(1), num.group(2), text, 1))
            else:
                center(d, text.upper(), new_page=True, after=LINE)
                self.toc.append(("", text, text, 0))
            return
        num = re.match(r"^(\S+) (.+)$", text)
        before = 0 if prev_kind and prev_kind.startswith("h") else 2 * LINE.pt
        para(d, text, bold=True, before=before, after=LINE, keep=True, align=WD_ALIGN_PARAGRAPH.LEFT)
        self.toc.append((num.group(1), num.group(2), text, 2 if kind == "h2" else 3))

    def body_file(self, path):
        d, prev = self.doc, None
        for kind, payload in parse(path.read_text(encoding="utf-8")):
            if kind.startswith("h"):
                self.heading(kind, payload, prev)
            elif kind == "p":
                para(d, self.cit.render(payload))
            elif kind in ("ul", "ol"):
                for marker, text in payload:
                    p = para(d, ("–" if kind == "ul" else marker) + "\t" + self.cit.render(text),
                             indent=Cm(-0.6), left=INDENT + Cm(0.6))
                    p.paragraph_format.tab_stops.add_tab_stop(INDENT + Cm(0.6))
            elif kind == "formula":
                center(d, payload, italic=True, before=LINE, after=LINE)
            elif kind == "tcap":
                self.table_count += 1
                para(d, payload, before=LINE, keep=True, align=WD_ALIGN_PARAGRAPH.LEFT)
            elif kind == "table":
                ncols = max(len(r) for r in payload)
                lens = [max(min(len(r[c]) if c < len(r) else 0, 60) for r in payload) + 6 for c in range(ncols)]
                widths = [TEXT_WIDTH_CM * n / sum(lens) for n in lens]
                make_table(d, [[self.cit.render(c) for c in r] for r in payload], widths, size=12)
                para(d, "", after=0)
            elif kind == "fig":
                caption, src = payload
                self.fig_count += 1
                p = center(d, before=LINE, keep=True)
                p.add_run().add_picture(str((HERE / src).resolve()), width=Cm(15))
                center(d, caption, after=LINE)
            prev = kind

    def references(self):
        d = self.doc
        title = "СПИСОК ИСПОЛЬЗОВАННОЙ ЛИТЕРАТУРЫ"
        center(d, title, new_page=True, after=LINE)
        self.toc.append(("", title, title, 0))
        for n, key in enumerate(self.cit.order, 1):
            para(d, f"[{n}] {self.cit.refs[key]}")

    def build(self):
        d = self.doc
        setup_section(d.sections[0])
        self.cover()
        sec = d.add_section(WD_SECTION.NEW_PAGE)
        setup_section(sec)
        sec.different_first_page_header_footer = True
        pg = OxmlElement("w:pgNumType")
        pg.set(qn("w:start"), "1")
        sec._sectPr.append(pg)
        page_number_footer(sec)
        d.sections[0].footer.is_linked_to_previous = False

        self.title_page()
        self.assignment()
        self.schedule()
        self.annotations()
        self.stats_apps = 0
        body = sorted(HERE.glob("[0-9][0-9]-*.md"))
        # содержание собирается после основной части (нужен список заголовков) и переносится к маркеру
        main, appendices = [p for p in body if not p.name.startswith("9")], [p for p in body if p.name.startswith("9")]
        marker = d.add_paragraph()
        for p in main:
            self.body_file(p)
        self.references()
        for p in appendices:
            self.body_file(p)
        # содержание вставляется перед основной частью
        toc_table = self.contents()
        body_el = d.element.body
        # последние элементы: заголовок «СОДЕРЖАНИЕ» (абзац) и таблица — переносим к маркеру
        heading_el = toc_table._tbl.getprevious()
        marker._p.addprevious(heading_el)
        marker._p.addprevious(toc_table._tbl)
        body_el.remove(marker._p)
        self.stats.update(figures=self.fig_count, tables=self.table_count, refs=len(self.cit.order), apps=self.stats_apps)
        OUT.mkdir(exist_ok=True)
        out = OUT / "diploma.docx"
        d.save(out)
        return out


def to_pdf(docx_path):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUT), str(docx_path)],
                   check=True, capture_output=True)
    return docx_path.with_suffix(".pdf")


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def locate(pdf_path, toc):
    """Номера страниц заголовков по напечатанному номеру страницы."""
    texts = [norm(p.extract_text() or "") for p in PdfReader(pdf_path).pages]
    title_idx = [i for i, t in enumerate(texts) if t.startswith("министерство")][1]
    numbers = [i - title_idx + 1 for i in range(len(texts))]
    start = next(i for i, t in enumerate(texts) if "содержание" in t[:20]) + 1
    pages = {}
    for _, _, key, _ in toc:
        k = norm(key)[:45]
        for idx in range(start, len(texts)):
            if k in texts[idx]:
                pages[key] = numbers[idx]
                start = idx
                break
    return pages, len(texts)


if __name__ == "__main__":
    b = Builder()
    pdf = to_pdf(b.build())
    pages, total = locate(pdf, b.toc)
    stats = dict(b.stats, pages=total - 1)
    b2 = Builder(pages, stats)
    pdf = to_pdf(b2.build())
    pages2, total2 = locate(pdf, b2.toc)
    missing = [k for _, _, k, _ in b2.toc if k not in pages2]
    moved = [k for k in pages2 if pages2[k] != pages.get(k)]
    print(f"{pdf}: {total2} стр.; не найдено: {missing}; сместились: {moved}; статистика: {b2.stats}", file=sys.stderr)
