import pandas as pd
import PyPDF2
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tkinter import filedialog, Tk

# Download the stopwords 
nltk.download('stopwords')

# Sample Perfect Resume
resume_data = pd.DataFrame({
    'resume_text': [
        "Experienced software engineer with machine learning expertise.",
        "Marketing specialist with data analysis skills.",
        "Graphic designer with strong communication skills.",
        "Entry-level software developer with Python experience."
    ]
})

# Keywords to look for
keywords = ["software", "engineer", "machine learning", "python", "data analysis"]

# Read the text from applicants PDF file
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extract_text()
    return text

# Extract keywords from Resume
def extract_keywords(text):
    tokens = nltk.word_tokenize(text.lower())
    stopwords = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
    return filtered_tokens

# GUI selecting the applicants PDF file
def choose_pdf_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    return file_path

# Ask user to choose file
print("Please choose a PDF file containing the resume.")
pdf_file_path = choose_pdf_file()

# Read selected file
new_resume_text = read_pdf(pdf_file_path)

# Preprocess new resume
new_resume_keywords = extract_keywords(new_resume_text)

# Displaying Extacted Key words
print("\nExtracted Keywords from New Resume:")
print(new_resume_keywords)

# Preprocessing text data 
def preprocess_text(text):
    return text.lower()

resume_data['processed_text'] = resume_data['resume_text'].apply(preprocess_text)

# Feature Engineering 
vectorizer = CountVectorizer(binary=True, vocabulary=keywords)
X = vectorizer.fit_transform(resume_data['processed_text'])
y = X.sum(axis=1)  # Number of matching keywords in each resume

# Training the model 
model = LogisticRegression()
model.fit(X, y)

# Feature vector for the new resume
new_resume_processed = preprocess_text(new_resume_text)
new_X = vectorizer.transform([new_resume_processed])
num_keywords_matched = new_X.sum()

# Predict number of matching keywords
predicted_matches = model.predict(new_X)[0]

# Display result
print("\nNew Resume:")
print(new_resume_text)
print(f"\nNumber of Keywords Matched: {num_keywords_matched}")
print(f"Predicted Number of Matches: {predicted_matches}")
