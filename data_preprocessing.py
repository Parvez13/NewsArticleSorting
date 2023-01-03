import re


# Create a function 'preprocess_text' for preprocess text
def preprocess_text(sent):
    """

    Args:
     sent: Takes raw sentence as an argument.
    returns:
     Returns as a sentence which are lowered, no punctuations, numbers, character
     and multiple spaces

    """
    try:
        # Lowercase
        sentence = sent.lower()
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Zäöüß]', ' ', sentence)
        # Remove single character
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence
    except Exception as e:
        print(f'Error raised!!: {e}')
