"""
setup.py  —  Run once before launching the app.
    python setup.py
"""
import os, sys

print("=" * 48)
print("  NEURON — One-time Setup")
print("=" * 48)

print("\n[1/2] Generating dataset...")
os.makedirs("data", exist_ok=True)
exec(open("data/generate_dataset.py").read())

print("\n[2/2] Training ML model...")
import ml_model
ml_model.train()

print("\n✅  Setup complete!")
print("\nTo run the app:")
print("    streamlit run app.py")
print("\nMake sure Ollama is running:")
print("    ollama serve")
print("    ollama pull llama3.2")
