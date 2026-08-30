import os
from app import demo

# Ensure required directories exist before launch
os.makedirs("outputs", exist_ok=True)
os.makedirs("temp_chunks", exist_ok=True)

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        allowed_paths=["outputs", "temp_chunks"]
    )
