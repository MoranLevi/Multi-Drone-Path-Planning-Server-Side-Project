from http.server import HTTPServer, BaseHTTPRequestHandler
import json
# import urllib.parse
# import cgi

numbersOfDrones = -1
file = None

class requestHandler(BaseHTTPRequestHandler):
    def end_headers(self): # CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(requestHandler, self).end_headers()

    def do_OPTIONS(self):
      self.send_response(200)
      self.end_headers()
    
    def do_GET(self):
        if self.path.endswith('/optimal-targets-classification'):
            # targetFile = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('targetFile', None)
            # numberOfDrones = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('numberOfDrones', None)
            # print(numberOfDrones)
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(numbersOfDrones.encode())
        if self.path.endswith('/required-targets-classification'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('Hi!'.encode())

    def do_POST(self):
        global numbersOfDrones
        if self.path.endswith('/numberOfDronesData'):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            dataLength = int(self.headers["Content-Length"])
            data = self.rfile.read(dataLength)
            numbersOfDrones = json.loads(data).get("numberOfDrones")
            print(numbersOfDrones)
            self.wfile.write('OK'.encode())

def main():
    PORT = 8000
    server = HTTPServer(('', PORT), requestHandler)
    print('Server running on port %s' % PORT)
    server.serve_forever()

if __name__ == '__main__':
    main()