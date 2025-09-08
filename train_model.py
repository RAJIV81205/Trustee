#!/usr/bin/env python3
"""
Script to train the internship matching model (run only once)
"""

from internship_matcher import InternshipMatchingModel
import os

def main():
    print("🚀 Internship Matching Model Training")
    print("=" * 40)
    
    # Check if model already exists
    model_files = ["internship_model.joblib", "tfidf_vectorizer.joblib", "scaler.joblib"]
    if all(os.path.exists(f) for f in model_files):
        print("⚠️  Model files already exist!")
        choice = input("Do you want to retrain? (y/N): ").lower().strip()
        if choice != 'y':
            print("✅ Using existing model. Run 'python production_main.py' to start API.")
            return
    
    # Initialize and train the model
    model = InternshipMatchingModel()
    model.train_model()
    
    print("=" * 40)
    print("✅ Training completed!")
    print("\nModel files created:")
    for file in model_files:
        print(f"- {file}")
    print("\n🚀 Run: python production_main.py")

if __name__ == "__main__":
    main()