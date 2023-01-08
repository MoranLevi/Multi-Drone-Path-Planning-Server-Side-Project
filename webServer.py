from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

class requestHandler(BaseHTTPRequestHandler):
    def end_headers(self): # CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(requestHandler, self).end_headers()

    def do_GET(self):
        if self.path.endswith('/optimal-targets-classification'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('Hello, world!'.encode())

def main():
    PORT = 8000
    server = HTTPServer(('', PORT), requestHandler)
    print('Server running on port %s' % PORT)
    server.serve_forever()

if __name__ == '__main__':
    main()