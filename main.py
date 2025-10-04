import Module.train as tr

if __name__ == "__main__":
    trainer = tr.Trainer()
    trainer.train_with_gradient_descent(epochs=100000)
    accuracy = trainer.evaluate()
    print(f"Test Accuracy: {accuracy:.4f}")
    trainer.plot_J_W()


'''
Diabetes_Detector/
│   main.py                 # Entry point for running the project
│   README.md               # Project description
│   requirements.txt        # Optional: list of dependencies
│
├───Module/                 # Python package for core ML code
│   ├───__init__.py
│   └───trainer.py          # Trainer class with gradient descent
│
├───Data/                   # Data-related scripts & datasets
│   ├───diabetes_dataset.csv
│   └───data_extractor.py   # Any preprocessing scripts
│
├───Visualization/          # All plotting / visualization scripts
│   ├───__init__.py
│   └───plotter.py
│
└───.idea/                  # IDE files (optional, usually ignored in git)


'''