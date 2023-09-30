import datetime
import logging

from astropy.coordinates import Angle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle
from os import path
from uncertainties import ufloat
from astropy import units as u
from astropy.time import Time
from dateutil.parser import parse


width, height = A4
resources_dir = path.join(path.dirname(__file__))


class ObservationReport:
    """
    Used to create a pdf file from the parameters and images generated in the plan stage.
    """
    LOGO_IMAGE = resources_dir + "/../resources/images/sherlock3.png"
    ALERT_IMAGE = resources_dir + "/resources/images/alert.png"

    def __init__(self, df_observatories, df, alert_date, object_id, name, working_path, ra, dec, t0, t0_low_err, t0_up_err, period,
                 period_low_err, period_up_err, duration, duration_low_err, duration_up_err, depth, depth_low_err,
                 depth_up_err, observable, min_dist, max_dist, min_altitude, max_days, v, j, h, k):
        self.df = df
        self.df_observatories = df_observatories.drop('tz', 1)
        self.alert_date = alert_date
        self.object_id = object_id
        self.name = name
        self.working_path = working_path
        self.images_path = working_path + "/images/"
        self.observable = observable
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.ra = ra
        self.dec = dec
        self.t0 = t0
        self.t0_low_err = t0_low_err
        self.t0_up_err = t0_up_err
        self.period = period
        self.period_low_err = period_low_err
        self.period_up_err = period_up_err
        self.duration = duration
        self.duration_low_err = duration_low_err
        self.duration_up_err = duration_up_err
        self.depth = depth
        self.depth_low_err = depth_low_err
        self.depth_up_err = depth_up_err
        self.min_altitude = min_altitude
        self.max_days = max_days
        self.v = v
        self.j = j
        self.h = h
        self.k = k

    def df_manipulations(self):
        """
        Performs changes of data from the initial dataframe for better formatting in the report.

        :return: the final dataframe
        """
        self.df['Observatory'] = self.df['observatory'].str.replace("-", " ")
        self.df['TZ'] = self.df['timezone'].fillna(0).astype('int')
        self.df['TZ'] = self.df['TZ'].astype('str')
        self.df['timezone'].astype(str)
        # Removing milliseconds from dates:
        self.df.loc[(self.df["ingress"].apply(lambda x: len(str(x)) > 4)), 'ingress'] = \
            self.df[(self.df["ingress"].apply(lambda x: len(str(x)) > 4))]['ingress'].str[:-4]
        self.df.loc[(self.df["twilight_evening"].apply(lambda x: len(str(x)) > 4)), 'twilight_evening'] = \
            self.df[(self.df["twilight_evening"].apply(lambda x: len(str(x)) > 4))]['twilight_evening'].str[:-4]
        self.df.loc[(self.df["midtime"].apply(lambda x: len(str(x)) > 4)), 'midtime'] = \
            self.df[(self.df["midtime"].apply(lambda x: len(str(x)) > 4))]['midtime'].str[:-4]
        self.df.loc[(self.df["egress"].apply(lambda x: len(str(x)) > 4)), 'egress'] = \
            self.df[(self.df["egress"].apply(lambda x: len(str(x)) > 4))]['egress'].str[:-4]
        self.df.loc[(self.df["twilight_morning"].apply(lambda x: len(str(x)) > 4)), 'twilight_morning'] = \
            self.df[(self.df["twilight_morning"].apply(lambda x: len(str(x)) > 4))]['twilight_morning'].str[:-4]
        self.df.loc[(self.df["start_obs"].apply(lambda x: len(str(x)) > 4)), 'start_obs'] = \
            self.df[(self.df["start_obs"].apply(lambda x: len(str(x)) > 4))]['start_obs'].str[:-4]
        self.df.loc[(self.df["end_obs"].apply(lambda x: len(str(x)) > 4)), 'end_obs'] = \
            self.df[(self.df["end_obs"].apply(lambda x: len(str(x)) > 4))]['end_obs'].str[:-4]
        self.df.loc[(self.df["ingress"].apply(lambda x: len(str(x)) <= 4)), 'ingress'] = '--'
        self.df.loc[(self.df["midtime"].apply(lambda x: len(str(x)) <= 4)), 'midtime'] = '--'
        self.df.loc[(self.df["egress"].apply(lambda x: len(str(x)) <= 4)), 'egress'] = '--'
        self.df.loc[(self.df["twilight_evening"].apply(lambda x: len(str(x)) <= 4)), 'twilight_evening'] = '--'
        self.df.loc[(self.df["twilight_morning"].apply(lambda x: len(str(x)) <= 4)), 'twilight_morning'] = '--'
        self.df.loc[(self.df["start_obs"].apply(lambda x: len(str(x)) <= 4)), 'start_obs'] = '--'
        self.df.loc[(self.df["end_obs"].apply(lambda x: len(str(x)) <= 4)), 'end_obs'] = '--'
        # Generamos la columna que contendrá el nombre de las imágenes:
        self.df['image_path'] = self.images_path + self.df['observatory'] + "_" + self.df['midtime'] + ".png"
        # Finalmente con las rutas creamos los objetos:
        self.df['Image'] = self.df['image_path'].apply(lambda x: Image(x))


        # We want to sort the dates row-wise, only keeping the interesting columns
        df_dates = self.df[['twilight_evening', 'ingress', 'midtime', 'egress', 'start_obs', 'end_obs', 'twilight_morning', 'timezone']]
        df_dates.columns = ['TWE', 'I', 'M', 'E', 'SO', 'EO', 'TWM', 'tz']
        dict_fechas = df_dates.to_dict('index')
        new_dict_fechas = {}
        for index, row in df_dates.iterrows():
            # Lo ordenamos por los valores de cada key:
            items = row[['TWE', 'I', 'M', 'E', 'SO', 'EO', 'TWM']].to_dict()
            list_of_tuples = sorted(items.items(), key=lambda item: item[1])
            # Definimos la noche como false:
            night = 0
            # Recorremos las tuplas y si noche es uno (lo que sucede a partir del TWE) ponemos en negrita la fecha.
            # Con esto vamos a generar una nueva lista de tuplas que tendrá las fechas apropiadas en negrita:
            list_of_tuples_with_night = []
            for tup in list_of_tuples:
                try:
                    y = list(tup)
                    local_time = parse(tup[1])
                    utime = Time(local_time)
                    y[1] = str(tup[1]) + ' (' + str(round(utime.jd, 2)) + ' JD)'
                    tup = tuple(y)
                except Exception as e:
                    logging.exception(f"Can't convert to date {tup[1]}")
                if night == 1 and tup[0] != 'TWM':
                    y = list(tup)
                    y[1] = '<strong>' + str(tup[1]) + '</strong>'
                    tup = tuple(y)
                if tup[0] == 'TWM':
                    night = 0
                if tup[0] == 'TWE':
                    night = 1
                list_of_tuples_with_night.append(tup)
            # Unimos las tuplas separando key y value con ":":
            join_list = [': '.join(tup) for tup in list_of_tuples_with_night]
            # Para terminar, el nuevo diccionario une los elementos de la lista separando por saltos de línea:
            new_dict_fechas[index] = '<font name="HELVETICA" size="8">' + '<br/>'.join(join_list) + '</font>'
        # Finalmente, creamos la columna Event times mapeando el index del dataframe con el del diccionario:
        self.df['Event times'] = self.df.index.to_series().map(new_dict_fechas)
        # La columna Transit Times Error será el concatenado de los errores:
        self.df['TT Error'] = '-' + self.df['midtime_low_err_h'].map(str) + '<br/>+' + self.df['midtime_up_err_h'].map(
            str)
        # La columna Moon será el concatenado de la moon_phase y de moon_dist:
        self.df['moon_phase'] = (self.df['moon_phase'] * 100).astype(int)
        self.df['moon_dist'] = self.df['moon_dist'].astype(int)
        self.df['Moon'] = self.df['moon_phase'].map(str) + '%<br/>' + self.df['moon_dist'].map(str) + 'º'
        # El dataframe final solo tendrá unas pocas columnas del excel inicial:
        df_output = self.df[['Observatory', 'TZ', 'Event times', 'TT Error', 'Moon', 'Image']]
        return df_output

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
        """
        Initializes the common header for all the pages.

        :param canvas: the report canvas
        :param doc: the report document
        """
        canvas.saveState()
        canvas.drawImage(self.LOGO_IMAGE, x=0, y=26.8 * cm, height=2.7 * cm, width=2.7 * cm, preserveAspectRatio=True)
        object_id_text = 'Sherlock observation plan: %s' % self.object_id
        canvas.setFont(psfontname="Helvetica", size=12)
        canvas.drawRightString(x=12 * cm, y=28 * cm, text=object_id_text)
        if doc.page == 1:
            object_id_text = '%s OBSERVATION PLAN' % self.object_id
            canvas.setFont(psfontname="Helvetica-Bold", size=25)
            canvas.drawCentredString(x=10 * cm, y=25.5 * cm, text=object_id_text)
        report_date = datetime.datetime.now().strftime("%a, %d %B %Y, %H:%M:%S")
        report_date_text = '%s' % report_date
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 28 * cm, report_date_text)
        canvas.restoreState()

    def create_footer(self, canvas, doc):
        """
        Initializes the common footer for all the pages

        :param canvas: the report canvas
        :param doc: the report document
        """
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
        page = "Powered by Astropy, Astroplan and ReportLab"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7 * cm, 0.5 * cm, page)
        page = "Page %s" % doc.page
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 0.5 * cm, page)
        canvas.restoreState()

    def create_report(self):
        """
        Creates the final report with all the star parameters, the candidate information and the observation nights
        data, storing it in a pdf.
        """
        df_manipulated = self.df_manipulations()
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

        # Content:
        story = [Spacer(1, 75)]
        introduction = '<font name="HELVETICA" size="9">This document is generated by the SHERLOCK PIPEline \
        transiting candidates observation planner. The target star this document focuses on is the %s. The star \
        parameters are shown in Table 1. The proposed candidate parameters are those exposed in Table 1. The  \
        observables included in the Table 5 are obtained based on the constraints of Table 3 and the list of \
        observatories from Table 4.</font>' % self.object_id
        story.append(Paragraph(introduction, styles["ParagraphAlignJustify"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 1 con los parámetros:
        tabla1_data = [['RA (deg)', 'Dec (deg)', 'V (mag)', 'J (mag)', 'H (mag)', 'K (mag)'],
                       [Angle(self.ra, u.deg).to_string(unit=u.hourangle, sep=':', precision=2),
                        Angle(self.dec, u.deg).to_string(unit=u.deg, sep=':', precision=2),
                        round(self.v, 2), round(self.j, 2), round(self.h, 2), round(self.k, 2)]]
        table1_colwidth = [3.5 * cm, 3.5 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm]
        table1_number_rows = len(tabla1_data)
        tabla1 = Table(tabla1_data, table1_colwidth, table1_number_rows * [0.75 * cm])
        tabla1.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        ObservationReport.row_colors(tabla1_data, tabla1)
        story.append(tabla1)

        table1_descripcion = '<font name="HELVETICA" size="9"><strong>Table 1: </strong>\
                        The proposed target parameters.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 2 con los parámetros:
        tabla2_data = [['T0 (TBJD)', 'Period (d)', 'Duration (h)', 'Depth (ppt)'],
                        [ufloat(self.t0, self.t0_low_err, self.t0_up_err),
                        ufloat(self.period, self.period_low_err, self.period_up_err),
                        ufloat(self.duration, self.duration_low_err, self.duration_up_err),
                        ufloat(self.depth, self.depth_low_err, self.depth_up_err)]]
        table2_colwidth = [4 * cm, 4 * cm, 3.5 * cm, 3.5 * cm]
        table2_number_rows = len(tabla2_data)
        tabla2 = Table(tabla2_data, table2_colwidth, table2_number_rows * [0.75 * cm])
        tabla2.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        ObservationReport.row_colors(tabla2_data, tabla2)
        story.append(tabla2)

        table1_descripcion = '<font name="HELVETICA" size="9"><strong>Table 2: </strong>\
                The proposed candidate parameters.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 3 con los parámetros:
        tabla3_data = [['Observability', 'Days interval', 'Min altitude (º)', 'Min Moon Dist (º)', 'Max Moon Dist (º)'],
                       [self.observable, self.max_days, self.min_altitude, self.min_dist, self.max_dist]]
        table3_colwidth = [2.5 * cm, 2.5 * cm, 2.5 * cm, 3 * cm, 3 * cm]
        table3_number_rows = len(tabla3_data)
        tabla3 = Table(tabla3_data, table3_colwidth, table3_number_rows * [0.75 * cm])
        tabla3.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        ObservationReport.row_colors(tabla3_data, tabla3)
        story.append(tabla3)

        table3_descripcion = '<font name="HELVETICA" size="9"><strong>Table 3: </strong>\
                        The constraints used to calculate the observables. Observability can have the next values: ' \
                             '1 - Entire transit is required, ' \
                             '0.5 - Transit midtime and either ingress or egress at least are required,\n' \
                             '0.25 - Only ingress or egress are required. ' \
                             'Days interval is the maximum number of days to search for observable transits.' \
                             'Min altitude is the minimum altitude above the horizon to consider an observable ' \
                             'transit. Min Moon Dist is the minimum distance to be kept from the target to the new ' \
                             'moon. Max Moon Dist is the minimum distance to be kept from the target to the full ' \
                             'moon. </font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table3_descripcion, styles["ParagraphAlignCenter"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 4 con los observatorios:
        table4_colwidth = [3.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]
        table4_number_rows = len(self.df_observatories) + 1  # Sumamos 1 para tener en cuenta el header
        table4_header_row = [['Name', 'Latitude (º)', 'Longitude (º)', 'Altitude (m)']]
        table4_data = self.df_observatories.values.tolist()
        tabla4 = Table(table4_header_row + table4_data, table4_colwidth, table4_number_rows * [0.75 * cm])
        tabla4.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        ObservationReport.row_colors(self.df_observatories, tabla4)
        story.append(tabla4)

        table4_descripcion = '<font name="HELVETICA" size="9"><strong>Table 3: </strong>\
                List of observatories.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table4_descripcion, styles["ParagraphAlignCenter"]))

        if self.alert_date is not None:
            story.append(Spacer(1, 30))
            story.append(Image(self.ALERT_IMAGE, width=15, height=15))
            alert = '<font name="HELVETICA" size="9" color="red">The table 5 with the observations will only contain \
                    scheduled observation nights up to the %s. Since then, the error propagation gets too wide to be \
                    acceptable for the plan to be precise enough.</font>' % self.alert_date
            story.append(Paragraph(alert, styles["ParagraphAlignJustify"]))

        # Pasamos a la siguiente página:
        story.append(Spacer(1, 30))

        # Generamos la tabla 5 con los observables:
        table5_header_row = [df_manipulated.columns[:, ].values.astype(str).tolist()]
        table5_data = df_manipulated.values.tolist()
        table5_data2 = []
        for row in table5_data:
            row_counter = 1
            manipulated_data = []
            for cell in row:
                # La última de las columnas se corresponde con imágenes, la mantenemos:
                if row_counter == len(row):
                    manipulated_data.append(cell)
                else:
                    manipulated_data.append(Paragraph(cell, styles["ParagraphAlignCenter"]))
                row_counter = row_counter + 1
            table5_data2.append(manipulated_data)

        final_data = table5_header_row + table5_data2

        # Creates a table with variable number of row and width:
        table5_number_rows = len(final_data)
        table5_rowwidths = []
        for x in range(table5_number_rows):
            if x == 0:  # Table headers
                table5_rowwidths.append(0.2 * inch)
            else:
                table5_rowwidths.append(1.2 * inch)

        # Creates a table with 5 columns, variable width:
        table5_colwidths = [1.1 * inch, 0.3 * inch, 2.5 * inch, 0.6 * inch, 0.5 * inch, 2.2 * inch]
        tabla5 = Table(final_data, table5_colwidths, table5_rowwidths, repeatRows=1)
        tabla5.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        ObservationReport.row_colors(final_data, tabla5)

        story.append(tabla5)

        table5_descripcion = '<font name="HELVETICA" size="9"><strong>Table 5</strong>: Observables for the \
                computed candidate. <strong>TZ column</strong> represents the observatory time zone offset from \
                UTC for each transit event. <strong>TT errors column</strong> represents the uncertainty of the \
                ingress, midtime and egress times for each observable calculated from the T0 and Period uncertainties. \
                <strong>Moon column</strong> represents the moon phase and distance to the target star at the \
                time of the transit midtime. <strong>Image column</strong> represents a plot where the altitude \
                and air mass are plotted with a blue line. The transit midtime is plotted as a vertical black bar \
                together with the ingress and egress in orange lines. The red vertical bars represent the uncertainty \
                of the ingress (left) and the egress (right). The green floor of the graph represents \
                the limit of altitude / air mass for the event to be observable. The white background and the \
                gray background represent the daylight and night based on the nautical twilights.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table5_descripcion, styles["ParagraphAlignJustify"]))
        # Definimos el frame y el template sabiendo que un A4: 21 cm de ancho por 29,7 cm de altura
        global_frame = Frame(1.5 * cm, 1.1 * cm, 18 * cm, 25.4 * cm, id='normal', showBoundary=0)
        global_template = PageTemplate(id='UnaColumna', frames=global_frame,
                                       onPage=self.create_header, onPageEnd=self.create_footer)
        # Construimos el documento:
        doc = BaseDocTemplate(self.working_path + "/" + self.name + "_observation_plan.pdf", pagesize=A4,
                              rightMargin=40, leftMargin=40,
                              topMargin=95, bottomMargin=15,
                              pageTemplates=global_template)
        doc.build(story)
