import webbrowser
import uvicorn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8000/docs")  
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    

