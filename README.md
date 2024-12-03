# Poetic Text Generator

A simple recurrent neural network (RNN) using LSTM layers to generate Shakespearean-style text based on a corpus of Shakespeare's works. The model demonstrates text prediction by learning patterns and generating creative sequences.

## Features
- Generates text in a Shakespearean style.
- Allows adjustable creativity with a `temperature` parameter.
- Pretrained model available for quick text generation.

## Requirements
- Python 3.7 or later
- TensorFlow 2.x
- NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/afnaayshu/poeticTextGenerator.git
   cd poeticTextGenerator
   ```
2. Install the required libraries:
```bash
pip install -r requirements.txt
```
## Usage
1. Train the Model:
  - Run main.py to preprocess the data, build the model, and train it.
  - The trained model will be saved as textgenerator.model.keras.
    
2. Generate Text:
  - The script supports generating text with different creativity levels using the temperature parameter.
  - Adjust the temperature in main.py and run the script: python main.py

3. Sample Output: <br>
-----------0.2------------ <br>
 rounding <br>
'sicilia is a so-forth:' 'tis this thing the stread the stand the streads,<br>
and thou stand the most his lance and the endle.<br>
<br>
clarence:<br>
he hath see the strease the stand the bantage.<br>
camillo:<br>
i cannot the streath, that thou should think the house,<br>
that this be the streathe the streath, the cannot the ends<br>
and the streathe the se<br>
-----------0.4------------<br>
he wenches<br>
say is a gallimaufry of gambond the trives,<br>
and all be the strease the light of the heavy with their country.<br>
<br>
edward:<br>
and think the stand to the house of my hands.<br>
<br>
clarence:<br>
i cannot that thought the seace with this dead.<br>
<br>
polixenes:<br>
that i take the edward men from the endle<br>
and thou wilt thou wilt that he diston that say.<br>
<br>
k<br>
-----------0.6------------<br>
red hate;<br>
nor never by advised purpose my gnord,<br>
here seep the scappant for man.<br>
<br>
capulet:<br>
i see it confesty, this ears the strange, thou should the stread,<br>
and what let this his bid and words, thou stand.<br>
<br>
is my gones tome, and thing henry'd us,<br>
show the brother pard me and thou to the streath<br>
and if are that you spoke this thou, then th<br>

## Project Structure
poeticTextGenerator/
├── main.py            # Main script for training and text generation
├── README.md          # Project documentation
├── requirements.txt   # List of dependencies
└── textgenerator.model.keras  # Trained model (optional, ignored in .gitignore)

## Parameters
- SEQ_LENGTH: Length of input sequence for training.
- STEP_SIZE: Step size for creating sequences.
- TEMPERATURE: Controls randomness in text generation.

## Customization
- To train on a custom text file, replace the Shakespeare corpus link with your text file.
- Modify hyperparameters in the script to experiment with the model.
Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Shakespeare Dataset
- Inspired by TensorFlow's text generation tutorials.
