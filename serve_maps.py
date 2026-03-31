#!/usr/bin/env python3
"""Simple HTTP server to serve the map files"""
import http.server
import socketserver
import os
from pathlib import Path

os.chdir('/Users/brandonk28/milind')

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")

print("\n" + "="*70)
print("🗺️  MAP SERVER STARTED")
print("="*70)
print(f"\n📍 Open these links in your browser:")
print(f"\n   🌲 Walayar Forest Map:")
print(f"      http://localhost:{PORT}/walayar_forest_map.html")
print(f"\n   🌾 Crop Fields Map (10,013 fields):")
print(f"      http://localhost:{PORT}/crop_fields_map.html")
print(f"\n   📊 Water Time Series:")
print(f"      http://localhost:{PORT}/more_water_timeseries.png")
print(f"\n" + "-"*70)
print(f"Press Ctrl+C to stop the server")
print("="*70 + "\n")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped.")
