# import os
import datetime
import glob
import gzip
import os
import pathlib
import pickle
import re

import alexfitter
import batman
import foldedleastsquares
from PyPDF2 import PdfReader
from astropy.coordinates import Angle
from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator
from pdf2image import pdf2image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, \
    Image, Table, TableStyle, ListFlowable
from os import path
from astropy import units as u
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from uncertainties import ufloat

from sherlockpipe.system_stability.mr_forecast import MrForecast

width, height = A4
resources_dir = path.join(path.dirname(__file__))


def replace_sub(match):
    inner = match.group(1)
    # Return the reformatted date
    return f"<sub>{inner}</sub>"

def replace_mathrm(match):
    inner = match.group(1)
    # Return the reformatted date
    return f"{{{inner}}}"

class MoriartyReport:
    """
    This class creates a report for the single transits module of SHERLOCK.
    """
    LOGO_IMAGE = resources_dir + "/../resources/images/sherlock3.png"
    CANDIDATE_COLORS = ['firebrick', 'cornflowerblue', 'pink', 'limegreen', 'sandybrown', 'turquoise', 'violet']

    def __init__(self, data_dir, object_id, ra, dec, v, j, h, k, candidates_df):
        self.data_dir = data_dir
        self.object_id = object_id
        self.ra = ra
        self.dec = dec
        self.v = v
        self.j = j
        self.h = h
        self.k = k
        self.candidates_df = candidates_df

    @staticmethod
    def row_colors(df, table_object):
        data_len = len(df)
        for each in range(1, data_len + 1):
            if each % 2 == 1:
                bg_color = colors.whitesmoke
            else:
                bg_color = colors.lightgrey
            table_object.setStyle(TableStyle([('BACKGROUND', (0, each), (-1, each), bg_color)]))

    def create_header(self, canvas, doc):
        canvas.saveState()

        # Logo:
        canvas.drawImage(self.LOGO_IMAGE, x=1.5 * cm, y=26.8 * cm, height=2 * cm, width=2 * cm,
                         preserveAspectRatio=True)

        # Title:
        object_id_text = 'MORIARTY Search Report: %s' % self.object_id
        canvas.setFont(psfontname="Helvetica", size=12)
        canvas.drawRightString(x=14.5 * cm, y=27.5 * cm, text=object_id_text)
        if doc.page == 1:
            object_id_text = '%s SINGLE-TRANSITS SEARCH REPORT' % self.object_id
            canvas.setFont(psfontname="Helvetica-Bold", size=20)
            canvas.drawCentredString(x=11 * cm, y=25.5 * cm, text=object_id_text)

        # Report date:
        report_date = datetime.datetime.now().strftime("%a, %d %B %Y, %H:%M:%S")
        report_date_text = '%s' % report_date

        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 28 * cm, report_date_text)

        canvas.restoreState()

    def create_footer(self, canvas, doc):
        canvas.saveState()

        # if doc.page == 1:
        #     # Footer con superíndice:
        #     textobject = canvas.beginText()
        #     textobject.setTextOrigin(1.8 * cm, 2.1 * cm)
        #     textobject.setFont("Helvetica", 5)
        #     textobject.setRise(5)
        #     textobject.textOut('1 ')
        #     textobject.setRise(0)
        #     textobject.setFont("Helvetica", 7)
        #     pie_pagina = 'Three possible observability values are defined: 1 - Entire transit is required, ' \
        #                  '0.5 - Transit midtime and either ingress or egress at least are required,\n' \
        #                  '0.25 - Only ingress or egress are required, with moon constraints of % sº as minimum ' \
        #                  'distance for new moon and % sº as minimum distance for full moon\n' \
        #                  'and for the observatories listed in the Table 2.' % (self.min_dist, self.max_dist)
        #
        #     for line in pie_pagina.splitlines():
        #         textobject.textLine(line)
        #
        #     canvas.drawText(textobject)

        # Powered by:
        page = "Powered by MORIARTY & ReportLab"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7 * cm, 0.5 * cm, page)

        # Page:
        page = "Page %s" % doc.page
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 0.5 * cm, page)

        canvas.restoreState()

    def is_float(self, element: any) -> bool:
        # If you expect None to be passed:
        if element is None:
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False

    def create_report(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ParagraphAlignCenter", alignment=TA_CENTER))
        styles.add(ParagraphStyle(name="ParagraphAlignJustify", alignment=TA_JUSTIFY))
        styles.wordWrap = 'LTR'
        table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                  ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                                  ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                                  ('FONTSIZE', (0, 0), (-1, -1), 10),
                                  ])
        table_style_small = TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                                        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                                        ('FONTSIZE', (0, 0), (-1, -1), 7),
                                        ])
        story = [Spacer(1, 75)]
        introduction = '<font name="HELVETICA" size="9">This document is created by the SHERLOCK single-transits report generator (' \
                       '<a href="https://github.com/franpoz/SHERLOCK" color="blue">https://github.com/franpoz/SHERLOCK</a>) ' \
                       'and focuses on the target star %s.</font>' % self.object_id
        story.append(Paragraph(introduction, styles["ParagraphAlignJustify"]))
        story.append(Spacer(1, 30))
        tabla1_data = [['RA (deg)', 'Dec (deg)', 'V (mag)', 'J (mag)', 'H (mag)', 'K (mag)'],
                       [Angle(self.ra, u.deg).to_string(unit=u.hourangle, sep=':',
                                                        precision=2) if self.ra is not None else '-',
                        Angle(self.dec, u.deg).to_string(unit=u.deg, sep=':',
                                                         precision=2) if self.dec is not None else '-',
                        round(self.v, 2) if self.v is not None else '-',
                        round(self.j, 2) if self.j is not None else '-',
                        round(self.h, 2) if self.h is not None else '-',
                        round(self.k, 2) if self.k is not None else '-']]
        table1_colwidth = [3.5 * cm, 3.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm]
        table1_number_rows = len(tabla1_data)
        tabla1 = Table(tabla1_data, table1_colwidth, table1_number_rows * [0.75 * cm])
        tabla1.setStyle(table_style)
        story.append(tabla1)
        table1_descripcion = '<font name="HELVETICA" size="9"><strong>Table 1: </strong>\
                        The proposed target parameters.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 15))
        table = 2
        figure = 1
        files = list(glob.glob(f"{self.data_dir}/{self.object_id}_t0_*_focus.png"))
        pat = re.compile(
            rf"[a-zA-Z]+ [0-9]+_t0_(-?\d+(?:\.\d+)?)_focus\.png$"
        )
        story.append(Image(f"{self.data_dir}/{self.object_id}.png", width=14 * cm, height=10 * cm))
        descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
            figure) + ': </strong>Above, the complete light curve. Center, the MORIARTY scores spectrum. Bottom, the autocorrelation of MORIARTY scores showing periodicity.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))

        story.append(Spacer(1, 30))

        fit_df = pd.read_csv(f"{self.data_dir}/{self.object_id}_fit.csv")
        tabla1_data = [['T0 (TBJD)', 'Depth (ppt)', 'Dur. (h)', 'S/N', '1σ Min. P (d)', 'Mean P (d)', '1σ Max P (d)', 'Rp (Earth)']]
        # Generamos la tabla 2 con los parámetros:
        for index, row in fit_df.iterrows():
            tabla1_data.append([
                round(row['t0'], 3),
                ufloat(row['depth'] * 1000, row['depth_err'] * 1000).format(".3uP"),
                ufloat(row['duration(h)'], row['duration_err(h)']).format(".3uP"),
                round(row['snr'], 2),
                round(row['period_min'], 2),
                round(row['period'], 2),
                round(row['period_max'], 2),
                ufloat(row['rp'], row['rp_err']).format(".3uP")]
            )
        table1_colwidth = [2 * cm, 2.5 * cm, 2.5 * cm, 1 * cm, 2.5 * cm, 2 * cm, 2.5 * cm, 2 * cm]
        table1_number_rows = len(tabla1_data)
        tabla1 = Table(tabla1_data, table1_colwidth, table1_number_rows * [0.75 * cm])
        tabla1.setStyle(table_style)
        story.append(tabla1)
        table1_descripcion = ('<font name="HELVETICA" size="9"><strong>Table 2: </strong>\
                        The proposed single-transits box-shape best-fit parameters. The period is an estimation from'
                              'random sampling the priors using different geometric approaches.</font>')
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 30))
        story.append(PageBreak())
        figure = figure + 1
        for file in files:
            match = pat.match(os.path.basename(file))
            if match:
                t0 = float(match.group(1))
            story.append(Image(file, width=16 * cm, height=10 * cm))
            descripcion = ('<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Focused single transit view at T=' + str(round(t0, 3)) +
                           ('. The red points are the MORIARTY positives. '
                            'In orange, the binned datapoints are plotted. In red, the best box-shaped fast-fit.</font>'))
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
            story.append(
                Image(f"{self.data_dir}/{self.object_id}_t0_" + str(t0) + "_all.png", width=16 * cm, height=8 * cm))
            descripcion = ('<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Global single transit view at T=' + str(round(t0, 3)) +
                           '. The red points are those within the best box-shaped fast-fit.</font>')
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
            story.append(
                Image(f"{self.data_dir}/{self.object_id}_t0_" + str(t0) + "_st.png", width=14 * cm, height=22 * cm))
            descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Vetting plot for single transit at T=' + str(round(t0, 3)) + '.</font>'
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
        global_frame = Frame(1.5 * cm, 1.1 * cm, 18 * cm, 25.4 * cm, id='normal', showBoundary=0)
        global_template = PageTemplate(id='UnaColumna', frames=global_frame,
                                       onPage=self.create_header, onPageEnd=self.create_footer)
        doc = BaseDocTemplate(self.data_dir + "/" + self.object_id + ".pdf", pagesize=A4,
                              rightMargin=40, leftMargin=40,
                              topMargin=95, bottomMargin=15,
                              pageTemplates=global_template)
        doc.build(story)
