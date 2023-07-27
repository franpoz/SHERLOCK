# import os
import datetime
import gzip
import os
import pathlib
import pickle

import alexfitter
import batman
import foldedleastsquares
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
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from uncertainties import ufloat

width, height = A4
resources_dir = path.join(path.dirname(__file__))


class FitReport:
    """
    This class creates a fit report for the fitting module of SHERLOCK.
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
        object_id_text = 'SHERLOCK Bayesian Fit Report: %s' % self.object_id
        canvas.setFont(psfontname="Helvetica", size=12)
        canvas.drawRightString(x=14.5 * cm, y=27.5 * cm, text=object_id_text)
        if doc.page == 1:
            object_id_text = '%s BAYESIAN FIT REPORT' % self.object_id
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
        page = "Powered by Allesfitter & ReportLab"
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
        f = gzip.GzipFile(os.path.join(self.data_dir, 'results/save_ns.pickle.gz'), 'rb')
        allesfitter_results = pickle.load(f)
        alles = alexfitter.allesclass(self.data_dir)
        inst = 'lc'
        key = 'flux'
        time = alles.data[inst]['time']
        flux = alles.data[inst][key]
        flux_err = alles.data[inst]['err_scales_' + key] * alles.posterior_params_median['err_' + key + '_' + inst]
        ns_table_df = pd.read_csv(self.data_dir + '/results/ns_table.csv')
        ns_derived_df = pd.read_csv(self.data_dir + '/results/ns_derived_table.csv')
        star_df = pd.read_csv(self.data_dir + '/params_star.csv')
        star_radius = star_df['R_star'].iloc[0]
        ld_a = ns_derived_df.loc[ns_derived_df['#property'] == 'Limb darkening; $u_\\mathrm{1; lc}$', 'value'].iloc[0]
        ld_b = ns_derived_df.loc[ns_derived_df['#property'] == 'Limb darkening; $u_\\mathrm{2; lc}$', 'value'].iloc[0]
        color_index = 0
        for companion in alles.settings['companions_phot']:
            color_index = color_index % len(self.CANDIDATE_COLORS)
            period = ns_table_df.loc[ns_table_df['#name'] == companion + '_period', 'median'].iloc[0]
            epoch = ns_table_df.loc[ns_table_df['#name'] == companion + '_epoch', 'median'].iloc[0]
            depth = ns_derived_df.loc[ns_derived_df[
                                          '#property'] == 'Transit depth (undil.) ' + companion + '; $\\delta_\\mathrm{tr; undil; ' + companion + '; lc}$ (ppt)', 'value'].iloc[
                0]
            total_duration = ns_derived_df.loc[ns_derived_df[
                                                   '#property'] == 'Total transit duration ' + companion + '; $T_\\mathrm{tot;' + companion + '}$ (h)', 'value'].iloc[
                0]
            inclination = ns_derived_df.loc[ns_derived_df[
                                                '#property'] == 'Inclination ' + companion + '; $i_\\mathrm{' + companion + '}$ (deg)', 'value'].iloc[
                0]
            semi_a = ns_derived_df.loc[ns_derived_df[
                                           '#property'] == 'Semi-major axis ' + companion + ' over host radius; $a_\\mathrm{' + companion + '}/R_\\star$', 'value'].iloc[
                0]
            radius_earth = ns_derived_df.loc[ns_derived_df[
                                                 '#property'] == 'Companion radius ' + companion + '; $R_\\mathrm{' + companion + '}$ ($\\mathrm{R_{\\oplus}}$)', 'value'].iloc[
                0]
            radius_star_units = ((radius_earth * u.R_earth).to(u.R_sun) / (star_radius * u.R_sun)).value
            params = batman.TransitParams()  # object to store transit parameters
            params.t0 = epoch  # time of inferior conjunction
            params.per = period  # orbital period
            params.rp = radius_star_units  # planet radius (in units of stellar radii)
            params.a = semi_a  # semi-major axis (in units of stellar radii)
            params.inc = inclination  # orbital inclination (in degrees)
            params.ecc = 0.  # eccentricity
            params.w = 90.  # longitude of periastron (in degrees)
            params.limb_dark = "quadratic"  # limb darkening model
            params.u = [ld_a, ld_b]  # limb da  # times at which to calculate light curve
            m = batman.TransitModel(params, time)  # initializes model
            model = m.light_curve(params)
            total_duration_over_period = (total_duration / 24) / period
            time_folded = foldedleastsquares.fold(time, period, epoch + period / 2)
            data_df = pd.DataFrame(columns=['time', 'time_folded', 'flux', 'model'])
            data_df['time'] = time
            data_df['time_folded'] = time_folded
            data_df['flux'] = flux
            data_df['model'] = model
            data_df = data_df.loc[(data_df['time_folded'] > 0.5 - total_duration_over_period * 3) & (
                    data_df['time_folded'] < 0.5 + total_duration_over_period * 3)]
            time_sub = data_df['time'].to_numpy()
            time_folded_sub = data_df['time_folded'].to_numpy()
            flux_sub = data_df['flux'].to_numpy()
            model_sub = data_df['model'].to_numpy()
            data_df = data_df.sort_values(by=['time_folded'], ascending=True)
            bin_means, bin_edges, binnumber = binned_statistic(time_folded_sub, flux_sub, statistic='mean', bins=40)
            bin_stds, _, _ = binned_statistic(time_folded_sub, flux_sub, statistic='std', bins=40)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            time_binned = bin_centers
            flux_binned = bin_means
            bin_means, bin_edges, binnumber = binned_statistic(time_folded_sub, model_sub, statistic='mean', bins=40)
            model_binned = bin_means
            color = self.CANDIDATE_COLORS[color_index]
            fig1 = plt.figure(1)
            fig1.patch.set_facecolor('xkcd:white')
            frame1 = fig1.add_axes((.1, .3, .8, .6))
            data_df_time_folded_hours = (data_df['time_folded'].to_numpy() * period - (period / 2)) * 24
            time_binned_hours = (time_binned * period - (period / 2)) * 24
            plt.scatter(data_df_time_folded_hours, data_df['flux'].to_numpy(),
                        color='gray', s=1, alpha=0.25, rasterized=True)
            plt.errorbar(time_binned_hours, flux_binned, yerr=bin_stds / 2,
                         marker='o', markersize=6, color=color, alpha=1, markeredgecolor='black', ls='none')
            plt.plot(data_df_time_folded_hours, data_df['model'].to_numpy(), color='black', linestyle='-', alpha=1)
            plt.xlim([-total_duration * 1.5, total_duration * 1.5])
            ylim = np.nanmax([1 - np.abs(np.nanmin(flux_binned) - np.nanmax(bin_stds)), np.abs(np.nanmax(flux_binned) + np.nanmax(bin_stds)) - 1])
            plt.ylim([1 - ylim, 1 + ylim])
            plt.xticks([])
            plt.ylabel(r'Relative flux', fontsize='small')
            plt.title(r'$P= ' + str(np.round(period, 2)) + r'd ,R=' + str(
                np.round(radius_earth, 2)) + r' \mathrm{R_{\oplus}}$, Depth=' + str(np.round(depth, 2)) + ' ppts')
            frame2 = fig1.add_axes((.1, .1, .8, .2))
            residuals = data_df['flux'].to_numpy() - data_df['model'].to_numpy()
            residuals_model = flux_binned - model_binned
            plt.scatter(data_df_time_folded_hours, residuals, color='gray', s=1, alpha=0.25, rasterized=True)
            plt.errorbar(time_binned_hours, residuals_model, yerr=bin_stds / 2,
                         marker='o', markersize=6, color=color, alpha=1, markeredgecolor='black', ls='none')
            ylim = np.nanmax([np.nanmin(residuals_model) - np.nanmax(bin_stds), np.abs(np.nanmax(residuals_model) + np.nanmax(bin_stds))])
            plt.ylim([-ylim, ylim])
            plt.xlim([-total_duration * 1.5, total_duration * 1.5])
            plt.xlabel(r"Time from mid-transit (hours)", fontsize='small')
            plt.ylabel(r'Relative flux', fontsize='small')
            plt.savefig(self.data_dir + '/' + companion + '_folded_curve.png', bbox_inches="tight")
            plt.close()
            color_index = color_index + 1
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
                                  ('FONTSIZE', (0, 0), (-1, -1), 7  ),
                                  ])
        # Content:
        story = [Spacer(1, 75)]
        introduction = '<font name="HELVETICA" size="9">This document is created by the SHERLOCK bayesian fit report generator (' \
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
        story.append(tabla1)
        table1_descripcion = '<font name="HELVETICA" size="9"><strong>Table 1: </strong>\
                        The proposed target parameters.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 15))
        # Generamos la tabla 2 con los parámetros:
        tabla2_data = [['Name', 'T0 (d)', 'Period (d)', 'Duration (h)', 'Depth (ppt)']]
        for index, candidate_row in self.candidates_df.iterrows():
            companion = candidate_row['name']
            tabla2_data = tabla2_data +\
                          [[companion,
                            round(candidate_row['t0'], 4),
                            round(candidate_row['period'], 4),
                            round(candidate_row['duration'] / 60, 2),
                            round(candidate_row['depth'], 3)]]
        table2_colwidth = [4 * cm, 4 * cm, 3.5 * cm, 3.5 * cm]
        table2_number_rows = len(tabla2_data)
        tabla2 = Table(tabla2_data, table2_colwidth, table2_number_rows * [0.75 * cm])
        tabla2.setStyle(table_style)
        story.append(tabla2)
        story.append(Spacer(1, 5))
        table2_descripcion = '<font name="HELVETICA" size="9"><strong>Table 2: </strong>' \
                             'The planetary signals parameters to be included in the bayesian fit.</font>'
        story.append(Paragraph(table2_descripcion, styles["ParagraphAlignCenter"]))
        story.append(Spacer(1, 15))
        table = 3
        figure = 1
        tabla_data = [['Type', 'Property', 'Value']]
        logz_arg = np.argmax(allesfitter_results.logz)
        logz = allesfitter_results.logz[logz_arg]
        logz_err = allesfitter_results.logzerr[logz_arg]
        tabla_data = tabla_data + [['Metric', 'Bayesian evidence (logZ)', ufloat(logz, logz_err, logz_err)]]
        for index, ns_row in ns_table_df.iterrows():
            if not any(companion in ns_row['#name'] for companion in alles.settings['companions_phot']) and "#" not in ns_row['#name']:
                tabla_data = tabla_data + \
                             [['Prior', ns_row['#name'],
                               ufloat(ns_row['median'],
                                      float(ns_row['lower_error']) if self.is_float(ns_row['lower_error']) else 0,
                                      float(ns_row['upper_error']) if self.is_float(ns_row['upper_error']) else 0)
                               ]]
        for index, ns_row in ns_derived_df.iterrows():
            if not any(companion in ns_row['#property'] for companion in alles.settings['companions_phot']) and "#" not in ns_row['#property']:
                tabla_data = tabla_data + \
                             [['Posterior', ns_row['#property'],
                               ufloat(ns_row['value'],
                                      float(ns_row['lower_error']) if self.is_float(ns_row['lower_error']) else 0,
                                      float(ns_row['upper_error']) if self.is_float(ns_row['upper_error']) else 0)
                               ]]
        table_colwidth = [2 * cm, 11 * cm, 5 * cm]
        table_number_rows = len(tabla_data)
        tabla = Table(tabla_data, table_colwidth, table_number_rows * [0.4 * cm])
        tabla.setStyle(table_style_small)
        story.append(tabla)
        story.append(Spacer(1, 5))
        table2_descripcion = '<font name="HELVETICA" size="9"><strong>Table ' + str(table) + ': </strong>' \
                              'The bayesian global priors and posteriors.</font>'
        story.append(Paragraph(table2_descripcion, styles["ParagraphAlignCenter"]))
        story.append(PageBreak())
        table = table + 1
        for index, candidate_row in self.candidates_df.iterrows():
            companion = candidate_row['name']
            folded_curve_file = self.data_dir + "/" + companion + "_folded_curve.png"
            if os.path.exists(folded_curve_file):
                story.append(Image(folded_curve_file, width=12 * cm, height=8 * cm))
                descripcion = '<font name="HELVETICA" size="9"><strong>Figure ' + str(
                    figure) + ': </strong>Folded curve with residuals of the final fit for ' + companion + '.</font>'
                story.append(Spacer(1, 5))
                story.append(Paragraph(descripcion, styles["ParagraphAlignCenter"]))
                story.append(Spacer(1, 15))
                figure = figure + 1
            tabla_data = [['Type', 'Property', 'Value']]
            for index, ns_row in ns_table_df.iterrows():
                if companion in ns_row['#name']:
                    tabla_data = tabla_data + \
                                 [['Prior', ns_row['#name'],
                                   ufloat(ns_row['median'],
                                          float(ns_row['lower_error']) if self.is_float(ns_row['lower_error']) else 0,
                                          float(ns_row['lower_error']) if self.is_float(ns_row['upper_error']) else 0)
                                   ]]
            for index, ns_row in ns_derived_df.iterrows():
                if companion in ns_row['#property']:
                    tabla_data = tabla_data + \
                                  [['Posterior', ns_row['#property'],
                                    ufloat(ns_row['value'],
                                          float(ns_row['lower_error']) if self.is_float(ns_row['lower_error']) else 0,
                                          float(ns_row['lower_error']) if self.is_float(ns_row['upper_error']) else 0)
                                    ]]
            table_colwidth = [2 * cm, 11 * cm, 5 * cm]
            table_number_rows = len(tabla_data)
            tabla = Table(tabla_data, table_colwidth, table_number_rows * [0.4 * cm])
            tabla.setStyle(table_style_small)
            story.append(tabla)
            story.append(Spacer(1, 5))
            table2_descripcion = '<font name="HELVETICA" size="9"><strong>Table ' + str(table) + ': </strong>' \
                                 'The bayesian fitted priors and posteriors for ' + candidate_row['name'] + '.</font>'
            story.append(Paragraph(table2_descripcion, styles["ParagraphAlignCenter"]))
            story.append(PageBreak())
            table = table + 1
        # Construimos el documento:
        global_frame = Frame(1.5 * cm, 1.1 * cm, 18 * cm, 25.4 * cm, id='normal', showBoundary=0)
        global_template = PageTemplate(id='UnaColumna', frames=global_frame,
                                       onPage=self.create_header, onPageEnd=self.create_footer)
        doc = BaseDocTemplate(self.data_dir + "/" + self.object_id + "_fit.pdf", pagesize=A4,
                              rightMargin=40, leftMargin=40,
                              topMargin=95, bottomMargin=15,
                              pageTemplates=global_template)
        doc.build(story)
        for file in os.listdir(self.data_dir):
            if ".png" in file:
                os.remove(self.data_dir + '/' + file)
