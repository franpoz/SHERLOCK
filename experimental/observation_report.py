# import os
import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
# from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, \
    Image, Table, TableStyle


width, height = A4


class ObservationReport:
    def __init__(self, df_observatories, df, object_id, images_path, logo_image,
                 t0, period, duration, depth, observable, min_dist, max_dist):
        self.df = df
        self.df_observatories = df_observatories
        self.object_id = object_id
        self.images_path = images_path
        self.logo_image = logo_image
        self.observable = observable
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.t0 = t0
        self.period = period
        self.duration = duration
        self.depth = depth

    def df_manipulations(self):
        self.df['Observatory'] = self.df['observatory'].str.replace("-", " ")

        # Generamos la columna que contendrá el nombre de las imágenes:
        self.df['image_path'] = self.images_path + self.df['observatory'] + "_" + self.df['midtime'] + ".png"
        # Reemplazamos : por _ para igualar la ruta al nombre de la imagen que se ha generado:
        self.df['image_path'] = self.df['image_path'].str.replace(":", "_")
        # Finalmente con las rutas creamos los objetos:
        self.df['Image'] = self.df['image_path'].apply(lambda x: Image(x))

        # Le quitamos los milisegundos a todos los campos de fecha:
        self.df['ingress'] = self.df['ingress'].str[:-4]
        self.df['twilight_evening'] = self.df['twilight_evening'].str[:-4]
        self.df['midtime'] = self.df['midtime'].str[:-4]
        self.df['egress'] = self.df['egress'].str[:-4]
        self.df['twilight_morning'] = self.df['twilight_morning'].str[:-4]

        # No vale con tener las fechas, las queremos ordenadas por filas. Nos quedamos con las columnas que necesitamos.
        df_fechas = self.df[['ingress', 'twilight_evening', 'midtime', 'egress', 'twilight_morning']]
        # Les ponemos el nombre definitivo:
        df_fechas.columns = ['I', 'TWE', 'M', 'E', 'TWM']
        # Convertimos el df en un diccionario manteniendo el index como key para ordenarlo.
        dict_fechas = df_fechas.to_dict('index')
        new_dict_fechas = {}
        # Recorremos el diccionario:
        for item in dict_fechas:
            # Lo ordenamos por los valores de cada key:
            list_of_tuples = sorted(dict_fechas[item].items(), key=lambda item: item[1])
            # Definimos la noche como false:
            night = 0
            # Recorremos las tuplas y si noche es uno (lo que sucede a partir del TWE) ponemos en negrita la fecha.
            # Con esto vamos a generar una nueva lista de tuplas que tendrá las fechas apropiadas en negrita:
            list_of_tuples_with_night = []
            for tup in list_of_tuples:
                if tup[0] == 'TWM':
                    night = 0
                if night == 1:
                    y = list(tup)
                    y[1] = '<strong>' + str(tup[1]) + '</strong>'
                    tup = tuple(y)
                if tup[0] == 'TWE':
                    night = 1
                list_of_tuples_with_night.append(tup)

            # Unimos las tuplas separando key y value con ":":
            join_list = [': '.join(tup) for tup in list_of_tuples_with_night]
            # Para terminar, el nuevo diccionario une los elementos de la lista separando por saltos de línea:
            new_dict_fechas[item] = '<br/>'.join(join_list)

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
        df_output = self.df[['Observatory', 'Event times', 'TT Error', 'Moon', 'Image']]

        return df_output

    @staticmethod
    def row_colors(df, table_object):
        data_len = len(df)
        for each in range(data_len):
            if each % 2 == 0:
                bg_color = colors.whitesmoke
            else:
                bg_color = colors.lightgrey

            table_object.setStyle(TableStyle([('BACKGROUND', (0, each), (-1, each), bg_color)]))

    def create_header(self, canvas, doc):
        canvas.saveState()

        # Logo:
        canvas.drawImage(self.logo_image, x=0, y=26.8 * cm, height=2.7 * cm, width=2.7 * cm)

        # Title:
        if doc.page > 1:
            object_id_text = 'Sherlock observation plan: %s' % self.object_id
            canvas.setFont(psfontname="Helvetica", size=12)
            canvas.drawRightString(x=10 * cm, y=28 * cm, text=object_id_text)
        else:
            object_id_text = '%s OBSERVATION PLAN' % self.object_id
            canvas.setFont(psfontname="Helvetica-Bold", size=25)
            canvas.drawCentredString(x=10 * cm, y=25.5 * cm, text=object_id_text)

        # Report date:
        report_date = datetime.datetime.now().strftime("%a, %d %B %Y, %H:%M:%S")
        report_date_text = '%s' % report_date

        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 28 * cm, report_date_text)

        canvas.restoreState()

    def create_footer(self, canvas, doc):
        canvas.saveState()

        if doc.page == 1:
            # Footer con superíndice:
            textobject = canvas.beginText()
            textobject.setTextOrigin(1.8 * cm, 2.1 * cm)
            textobject.setFont("Helvetica", 5)
            textobject.setRise(5)
            textobject.textOut('1 ')
            textobject.setRise(0)
            textobject.setFont("Helvetica", 7)
            pie_pagina = 'Three possible observability values are defined: 1 - Entire transit is required, ' \
                         '0.5 - Transit midtime and either ingress or egress at least are required,\n' \
                         '0.25 - Only ingress or egress are required, with moon constraints of % sº as minimum ' \
                         'distance for new moon and % sº as minimum distance for full moon\n' \
                         'and for the observatories listed in the Table 2.' % (self.min_dist, self.max_dist)

            for line in pie_pagina.splitlines():
                textobject.textLine(line)

            canvas.drawText(textobject)

        # Powered by:
        page = "Powered by Astropy, Astroplan and ReportLab"
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7 * cm, 0.5 * cm, page)

        # Page:
        page = "Page %s" % doc.page
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(20.5 * cm, 0.5 * cm, page)

        canvas.restoreState()

    def create_report(self, df_manipulated):
        # Definimos los estilos que vamos a usar:
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="ParagraphAlignCenter", alignment=TA_CENTER))
        styles.add(ParagraphStyle(name="ParagraphAlignJustify", alignment=TA_JUSTIFY))
        styles.wordWrap = 'LTR'
        table_style = TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                  ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                  ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                                  ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
                                  ])

        # Definimos el contenido:
        story = [Spacer(1, 75)]

        introduction = '<font name="HELVETICA" size="9">This document is generated by the SHERLOCK PIPEline \
        transiting candidates observation planner. The target star this document focuses on is the %s. The \
        proposed candidate parameters are those exposed in Table 1. The observables included in the Table 3 \
        are obtained based on a observability of %s </font>' % (self.object_id, self.observable)
        story.append(Paragraph(introduction, styles["ParagraphAlignJustify"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 1 con los parámetros:
        tabla1_data = [['T0 (TBJD)', 'Period (d)', 'Duration (min)', 'Depth (ppt)'],
                       [self.t0, self.period, self.duration, self.depth]]
        table1_colwidth = [2.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]
        table1_number_rows = len(tabla1_data)
        tabla1 = Table(tabla1_data, table1_colwidth, table1_number_rows * [0.75 * cm])
        tabla1.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        self.row_colors(tabla1_data, tabla1)
        story.append(tabla1)

        table1_descripcion = '<font name="HELVETICA" size="9"><strong>Table 1: </strong>\
                The proposed candidate parameters.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table1_descripcion, styles["ParagraphAlignCenter"]))

        story.append(Spacer(1, 30))

        # Generamos la tabla 2 con los observatorios:
        table2_colwidth = [3.5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm]
        table2_number_rows = len(self.df_observatories) + 1  # Sumamos 1 para tener en cuenta el header
        table2_header_row = [self.df_observatories.columns[:, ].values.astype(str).tolist()]
        table2_data = self.df_observatories.values.tolist()
        tabla2 = Table(table2_header_row + table2_data, table2_colwidth, table2_number_rows * [0.75 * cm])
        tabla2.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        self.row_colors(self.df_observatories, tabla2)
        story.append(tabla2)

        table2_descripcion = '<font name="HELVETICA" size="9"><strong>Table 2: </strong>\
                List of observatories.</font>'
        story.append(Spacer(1, 5))
        story.append(Paragraph(table2_descripcion, styles["ParagraphAlignCenter"]))

        # Pasamos a la siguiente página:
        story.append(PageBreak())

        # Generamos la tabla 3 con los observables:
        table3_header_row = [df_manipulated.columns[:, ].values.astype(str).tolist()]
        table3_data = df_manipulated.values.tolist()
        table3_data2 = []
        for row in table3_data:
            row_counter = 1
            manipulated_data = []
            for cell in row:
                # La última de las columnas se corresponde con imágenes, la mantenemos:
                if row_counter == len(row):
                    manipulated_data.append(cell)
                else:
                    manipulated_data.append(Paragraph(cell, styles["ParagraphAlignCenter"]))
                row_counter = row_counter + 1
            table3_data2.append(manipulated_data)

        final_data = table3_header_row + table3_data2

        # Creates a table with variable number of row and width:
        table3_number_rows = len(final_data)
        table3_rowwidths = []
        for x in range(table3_number_rows):
            if x == 0:  # Table headers
                table3_rowwidths.append(0.2 * inch)
            else:
                table3_rowwidths.append(1.2 * inch)

        # Creates a table with 5 columns, variable width:
        table3_colwidths = [1.1 * inch, 2.3 * inch, 0.7 * inch, 0.7 * inch, 2.2 * inch]
        tabla3 = Table(final_data, table3_colwidths, table3_rowwidths, repeatRows=1)
        tabla3.setStyle(table_style)
        # Le damos el estilo alternando colores de filas:
        self.row_colors(final_data, tabla3)

        story.append(tabla3)

        table3_descripcion = '<font name="HELVETICA" size="9"><strong>Table 3</strong>: Observables for the \
                computed candidate. <strong>TT errors column</strong> represents the uncertainty of the ingress, \
                midtime and egress times for each observable calculated from the T0 and Period uncertainties. \
                <strong>Moon column</strong> represents the moon phase and distance to the target star at the \
                time of the transit midtime. <strong>Image column</strong> represents a plot where the altitude \
                and air mass are plotted with a blue line. The transit midtime is plotted in the center of the \
                graph together with the ingress and egress in orange lines. The green floor of the graph represent \
                the limit of altitude / air mass for the event to be observable. The white background and the \
                gray background represent the daylight and night based on the nautical twilights.</font>'

        story.append(Spacer(1, 5))
        story.append(Paragraph(table3_descripcion, styles["ParagraphAlignJustify"]))

        # Definimos el frame y el template sabiendo que un A4: 21 cm de ancho por 29,7 cm de altura
        global_frame = Frame(1.5 * cm, 1.1 * cm, 18 * cm, 25.4 * cm, id='normal', showBoundary=0)

        global_template = PageTemplate(id='UnaColumna', frames=global_frame,
                                       onPage=self.create_header, onPageEnd=self.create_footer)

        # Construimos el documento:
        doc = BaseDocTemplate("form_letter.pdf", pagesize=A4,
                              rightMargin=40, leftMargin=40,
                              topMargin=95, bottomMargin=15,
                              pageTemplates=global_template)
        doc.build(story)


""" Esta parte sobra, se mantiene para que se vea como inicializar la clase y que necesita:
if __name__ == "__main__":
    t0_variable = 14
    period_variable = 5
    duration_variable = 120
    depth_variable = 1.5
    observable_variable = 'por definir'
    min_dist_variable = 5
    max_dist_variable = 15

    ruta = r"C:\Users\Fenix\Documents\GitHub\createPDF\plan\observation_plan.csv"
    df_csv = pd.read_csv(ruta)

    id_object = '123789'
    path_images = r'plan\\'
    path_logo_image = '..\images\sherlock2.png'

    test_observatories_data = [['Observatory 1', -3.265, 40.569, 200],
                               ['Observatory 2', 3.243, 42.791, 1200],
                               ['Observatory 3', 5.231, 30.632, 806],
                               ['Observatory 4', -1.549, 38.609, 456],
                               ['Observatory 5', -2.265, 39.720, 215],
                               ['Observatory 6', -3.265, 40.831, 949],
                               ['Observatory 7', -4.265, 41.942, 2695],
                               ['Observatory 8', -5.265, 42.053, 231],
                               ['Observatory 9', -6.265, 43.164, 2329],
                               ['Observatory 10', -7.265, 44.275, 132]]
    df_observatory = pd.DataFrame(test_observatories_data, columns=['Name', 'Latitude', 'Longitude', 'Altitude (m)'])

    # Inicializamos la clase:
    report = ObservationReport(df_observatory, df_csv, id_object, path_images, path_logo_image,
                               t0_variable, period_variable, duration_variable, depth_variable,
                               observable_variable, min_dist_variable, max_dist_variable)
    # Preparamos el df que formará la tabla de observables:
    df_final = report.df_manipulations()
    # Generamos el reporte.
    report.create_report(df_final)

    # Para abrir el PDF una vez sea generado:
    # os.system("form_letter.pdf")

"""
