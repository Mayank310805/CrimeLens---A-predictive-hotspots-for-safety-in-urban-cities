# report_generator.py
# This module contains the logic for generating the PDF analysis report.

from fpdf import FPDF
from datetime import datetime
import pandas as pd

class PDFReport(FPDF):
    """
    A class to generate a standardized PDF report for the CrimeLens analysis.
    """
    def header(self):
        """ Defines the header for each page of the report. """
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CrimeLens: Predictive Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """ Defines the footer for each page of the report. """
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_summary_section(self, total_crimes, date_range, crime_types):
        """ Adds a summary section with key statistics to the report. """
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, '1. Analysis Summary', 0, 1, 'L')
        
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, 
            f"- Total Crimes Analyzed: {total_crimes}\n"
            f"- Date Range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}\n"
            f"- Selected Crime Types: {', '.join(crime_types)}"
        )
        self.ln(5)

    def add_charts_section(self, hourly_chart_path, daily_chart_path):
        """ Adds the temporal analysis charts to the report. """
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, '2. Temporal Analysis', 0, 1, 'L')
        
        if hourly_chart_path:
            self.image(hourly_chart_path, x=10, w=190)
            self.ln(5)
        
        if daily_chart_path:
            self.image(daily_chart_path, x=10, w=190)
        self.ln(10)

    def add_hotspots_table_section(self, hotspot_data):
        """ Adds a table of the top 5 hotspots to the report. """
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, '3. Top 5 Crime Hotspots Detected', 0, 1, 'L')

        if hotspot_data.empty:
            self.set_font('Arial', '', 10)
            self.cell(0, 10, "No significant hotspots were detected with the current settings.", 0, 1, 'L')
            return

        top_hotspots = hotspot_data.groupby('cluster').size().nlargest(5).reset_index(name='crime_count')
        centroids = hotspot_data.groupby('cluster')[['latitude', 'longitude']].mean()
        top_hotspots = top_hotspots.merge(centroids, on='cluster')

        self.set_font('Arial', 'B', 10)
        self.cell(30, 10, 'Hotspot ID', 1, 0, 'C')
        self.cell(80, 10, 'Approx. Location (Lat, Lon)', 1, 0, 'C')
        self.cell(40, 10, 'Crime Count', 1, 1, 'C')

        self.set_font('Arial', '', 10)
        for _, row in top_hotspots.iterrows():
            self.cell(30, 10, str(int(row['cluster'])), 1, 0, 'C')
            self.cell(80, 10, f"{row['latitude']:.4f}, {row['longitude']:.4f}", 1, 0, 'C')
            self.cell(40, 10, str(row['crime_count']), 1, 1, 'C')

def create_report(filtered_data, hotspot_data, date_range, crime_types, hourly_chart_path, daily_chart_path):
    """
    Main function to orchestrate the creation of the PDF report.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.add_summary_section(
        total_crimes=len(filtered_data),
        date_range=date_range,
        crime_types=crime_types
    )
    
    pdf.add_charts_section(hourly_chart_path, daily_chart_path)
    
    pdf.add_hotspots_table_section(hotspot_data)
    
    # ✨ --- THE FINAL FIX --- ✨
    # Convert the 'bytearray' from FPDF into the strict 'bytes' format that Streamlit requires.
    return bytes(pdf.output(dest='S'))

