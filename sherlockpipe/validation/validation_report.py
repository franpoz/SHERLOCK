# import os
import datetime
import os
import pathlib

from PyPDF2 import PdfReader
from astropy.coordinates import Angle
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

width, height = A4
resources_dir = path.join(path.dirname(__file__))


class ValidationReport:
    """
    This class creates a validation report for the validation module of SHERLOCK.
    """
    LOGO_IMAGE = resources_dir + "/../resources/images/sherlock3.png"

    def __init__(self, data_dir, file_name, object_id, ra, dec, t0, period, duration, depth, v, j, h, k):
        self.data_dir = data_dir
        self.file_name = file_name
        self.object_id = object_id
        self.ra = ra
        self.dec = dec
        self.t0 = t0
        self.period = period
        self.duration = duration
        self.depth = depth
        self.v = v
        self.j = j
        self.h = h
        self.k = k

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
        object_id_text = 'SHERLOCK Statistical Validation Report: %s' % self.object_id
        canvas.setFont(psfontname="Helvetica", size=12)
        canvas.drawRightString(x=14.5 * cm, y=27.5 * cm, text=object_id_text)
        if doc.page == 1:
            object_id_text = '%s STATISTICAL VALIDATION REPORT' % self.object_id
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
        page = "Powered by TRICERATOPS & ReportLab"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7 * cm, 0.5 * cm, page)

        # Page:
        page = "Page %s" % doc.page
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 0.5 * cm, page)

        canvas.restoreState()

    def create_report(self):
        # Styles to be used
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
                                  ('FONTSIZE', (0, 0), (-1, -1), 8),
                                  ])
        # Content:
        story = [Spacer(1, 75)]
        introduction = '<font name="HELVETICA" size="9">This document is created by the SHERLOCK statistical validtion report generator (' \
                       '<a href="https://github.com/franpoz/SHERLOCK" color="blue">https://github.com/franpoz/SHERLOCK</a>) ' \
                       'and focuses on the target star %s.</font>' % self.object_id
        story.append(Paragraph(introduction, styles["ParagraphAlignJustify"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 1 con los parámetros:
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
        # Le damos el estilo alternando colores de filas:
        ValidationReport.row_colors(tabla1_data, tabla1)
        story.append(tabla1)
        table1_descripcion = '<font name="HELVETICA" size="9"><strong>Table 1: </strong>\
                        The proposed target parameters.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 15))
        # Generamos la tabla 2 con los parámetros:
        tabla2_data = [['T0 (d)', 'Period (d)', 'Duration (h)', 'Depth (ppt)'],
                       [round(self.t0, 4),
                        round(self.period, 4),
                        round(self.duration / 60, 2),
                        round(self.depth, 3)]]
        table2_colwidth = [4 * cm, 4 * cm, 3.5 * cm, 3.5 * cm]
        table2_number_rows = len(tabla2_data)
        tabla2 = Table(tabla2_data, table2_colwidth, table2_number_rows * [0.75 * cm])
        tabla2.setStyle(table_style)
        ValidationReport.row_colors(tabla2_data, tabla2)
        story.append(tabla2)
        story.append(Spacer(1, 5))
        table2_descripcion = '<font name="HELVETICA" size="9"><strong>Table 2: </strong>' \
                             'The candidate parameters.</font>'
        story.append(Paragraph(table2_descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 15))
        folded_curve_file = self.data_dir + "/folded_curve.png"
        figure = 1
        if os.path.exists(folded_curve_file):
            story.append(Image(folded_curve_file, width=16 * cm, height=10 * cm))
            descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Folded curve of the found candidate.</font>'
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
        contrast_curve_file = self.data_dir + "/contrast_curve.png"
        if os.path.exists(contrast_curve_file):
            story.append(Image(contrast_curve_file, width=16 * cm, height=10 * cm))
            descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Contrast curve for target.</font>'
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
        all_files = os.listdir(self.data_dir)
        fov_file = self.data_dir + '/fov.png'
        for file in all_files:
            if file.startswith('field_'):
                images = pdf2image.convert_from_path(self.data_dir + '/' + file)
                for i in range(len(images)):
                    # Save pages as images in the pdf
                    images[i].save(fov_file, 'PNG')
                break
        story.append(Image(fov_file, width=16 * cm, height=7 * cm))
        descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
            figure) + ': </strong>Nearby stars for target and its aperture</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 15))
        figure = figure + 1
        validation_file = self.data_dir + "/validation_scenarios.csv"
        if os.path.exists(validation_file):
            table_data = [['ID', 'scenario', 'M_s', 'R_s', 'P_orb', 'inc', 'b', 'ecc', 'w', 'R_p', 'M_EB', 'R_EB',
                           'prob']]
            metrics_df = pd.read_csv(validation_file)
            for index, metric_row in metrics_df.iterrows():
                table_data.append([str(metric_row['ID']),
                                   metric_row['scenario'],
                                   round(metric_row['M_s'], 2),
                                   round(metric_row['R_s'], 2),
                                   round(metric_row['P_orb'], 2),
                                   round(metric_row['inc'], 2),
                                   round(metric_row['b'], 2),
                                   round(metric_row['ecc'], 2),
                                   round(metric_row['w'], 2),
                                   round(metric_row['R_p'], 2),
                                   round(metric_row['M_EB'], 2),
                                   round(metric_row['R_EB'], 2),
                                   round(metric_row['prob'], 6)])
            table_colwidth = [2.3 * cm, 2 * cm, 1 * cm, 1 * cm, 1 * cm, 1 * cm, 1 * cm, 1 * cm, 1 * cm, 1 * cm,
                              1 * cm, 1 * cm, 2.5 * cm]
            table_number_rows = len(table_data)
            table = Table(table_data, table_colwidth, table_number_rows * [0.5 * cm])
            table.setStyle(table_style_small)
            ValidationReport.row_colors(metrics_df, table)
            story.append(table)
            story.append(Spacer(1, 5))
            table_descripcion = '<font name="HELVETICA" size="9"><strong>Table 3: </strong>' \
                                'Scenarios attributes and probabilities.</font>'
            story.append(Paragraph(table_descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
        validation_file = self.data_dir + "/validation.csv"
        if os.path.exists(validation_file):
            table_data = [['Scenario', 'FPP', 'NFPP', 'FPP2', 'FPP3+']]
            metrics_df = pd.read_csv(validation_file)
            for index, metric_row in metrics_df.iterrows():
                table_data.append([metric_row['scenario'],
                                   round(metric_row['FPP'], 6),
                                   round(metric_row['NFPP'], 6),
                                   round(metric_row['FPP2'], 6),
                                   round(metric_row['FPP3+'], 6)])
            table_colwidth = [4 * cm, 4 * cm, 3.5 * cm]
            table_number_rows = len(table_data)
            table = Table(table_data, table_colwidth, table_number_rows * [0.5 * cm])
            table.setStyle(table_style)
            ValidationReport.row_colors(metrics_df, table)
            story.append(table)
            story.append(Spacer(1, 5))
            table_descripcion = '<font name="HELVETICA" size="9"><strong>Table 4: </strong>' \
                                'Validation results.</font>'
            story.append(Paragraph(table_descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
        result_map_file = self.data_dir + "/triceratops_map.png"
        if os.path.exists(result_map_file):
            story.append(Image(result_map_file, width=9 * cm, height=9 * cm))
            descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Validation map for the mean scenario.</font>'
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
        scenario = 0
        scenarios_file = self.data_dir + "/scenario_" + str(scenario) + "_fits.pdf"
        scenarios_png_file = self.data_dir + "/scenario_" + str(scenario) + "_fits.png"
        while os.path.exists(scenarios_file):
            images = pdf2image.convert_from_path(scenarios_file)
            for i in range(len(images)):
                # Save pages as images in the pdf
                images[i].save(scenarios_png_file, 'PNG')
            story.append(Image(scenarios_png_file, width=16 * cm, height=20 * cm))
            descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
                figure) + ': </strong>Best fits for scenario no. .' + str(scenario) + '</font>'
            story.append(Spacer(1, 5))
            story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
            story.append(Spacer(1, 15))
            figure = figure + 1
            scenario = scenario + 1
            scenarios_file = self.data_dir + "/scenario_" + str(scenario) + "_fits.pdf"
            scenarios_png_file = self.data_dir + "/scenario_" + str(scenario) + "_fits.png"
        # neighbours_file_index = 0
        # neighbours_file = self.data_dir + "/star_nb_" + (neighbours_file_index) + ".png"
        # figure = 1
        # while os.path.exists(neighbours_file):
        #     story.append(Image(neighbours_file, width=16 * cm, height=16 * cm))
        #     descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(figure) + ': </strong>' \
        #                          'Nearby stars folded plot with the same period than the candidate.</font>'
        #     story.append(Spacer(1, 5))
        #     story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
        #     story.append(Spacer(1, 15))
        #     neighbours_file_index = neighbours_file_index + 1
        #     neighbours_file = self.data_dir + "/star_nb_" + str(neighbours_file_index) + ".png"
        #     figure = figure + 1

        # Construimos el documento:
        global_frame = Frame(1.5 * cm, 1.1 * cm, 18 * cm, 25.4 * cm, id='normal', showBoundary=0)
        global_template = PageTemplate(id='UnaColumna', frames=global_frame,
                                       onPage=self.create_header, onPageEnd=self.create_footer)
        doc = BaseDocTemplate(self.data_dir + "/" + self.object_id + "_" + self.file_name, pagesize=A4,
                              rightMargin=40, leftMargin=40,
                              topMargin=95, bottomMargin=15,
                              pageTemplates=global_template)
        doc.build(story)
