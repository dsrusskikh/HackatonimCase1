## LSTM

import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

df = pd.read_excel('CRA_train_1200.xlsx')
#df = df.head(100) # для быстрого теста работы кода
df

df['pr_txt'] = df['pr_txt'].str.lower()

# перекодируем категории
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['Уровень рейтинга'])

# разделим данные на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(df['pr_txt'], df['category_encoded'], test_size=0.2, random_state=42)

# конвертируем текстовые данные в tf-idf вектора для будущей обработки моделью
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# посмотрим матрицы
print("X_train_tfidf shape:", X_train_tfidf.shape)
print("X_test_tfidf shape:", X_test_tfidf.shape)

# зададим архитектуру модели
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Reshape x to (batch_size, sequence_length, input_dim)
        x = x.view(x.size(0), -1, x.size(1))

        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

# зададим гиперпараметры
input_dim = X_train_tfidf.shape[1]
hidden_dim = 128
output_dim = len(df['Уровень рейтинга'].unique())
n_layers = 2
dropout = 0.2

# зададим нагрузку на устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

# просмотр архитектуры
print(model)

# пропишем функцию потерь и оптимизатор
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# конвертируем данные в PyTorch tensors
X_train_tfidf = torch.FloatTensor(X_train_tfidf.toarray()).to(device)
y_train = torch.LongTensor(numpy.array(y_train)).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# тренируем модель
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tfidf)

    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# конвертируем данные в последовательность, а затем в тензор
y_test = torch.tensor(numpy.array(y_test), dtype=torch.long).to(device)
X_test_tfidf_dense = torch.FloatTensor(X_test_tfidf.toarray()).to(device)

model.eval()
with torch.no_grad():
    outputs = model(X_test_tfidf_dense)
    _, predicted = torch.max(outputs, 1)

correct = (predicted == y_test).sum().item()
total = y_test.size(0)
accuracy = correct / total * 100
print(f'Test Accuracy: {accuracy:.2f}%')
# оценка точности модели

w_f1_score = f1_score(y_test, predicted, average='weighted')
print(f'F1 Score: {w_f1_score:.4f}')

# сохраним модель
torch.save(model, 'model_LSTM.pth')

# та же модель, но для укрупненных категорий
## LSTM (для укрупненных значений)

df['category_encoded1'] = label_encoder.fit_transform(df['Категория'])

# разделим данные на обучающие и тестовые
X_train1, X_test1, y_train1, y_test1 = train_test_split(df['pr_txt'], df['category_encoded1'], test_size=0.2, random_state=42)

# конвертируем текстовые данные в tf-idf вектора для будущей обработки моделью
X_train_tfidf1 = tfidf_vectorizer.fit_transform(X_train1)
X_test_tfidf1 = tfidf_vectorizer.transform(X_test1)

# посмотрим матрицы
print("X_train_tfidf shape:", X_train_tfidf1.shape)
print("X_test_tfidf shape:", X_test_tfidf1.shape)

# зададим гиперпараметры
input_dim = X_train_tfidf1.shape[1]
hidden_dim = 128
output_dim = len(df['Категория'].unique())
n_layers = 2
dropout = 0.2

model1 = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

# просмотр архитектуры
print(model1)

# пропишем функцию потерь и оптимизатор
criterion = nn.NLLLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# конвертируем данные в PyTorch tensors
X_train_tfidf1 = torch.FloatTensor(X_train_tfidf1.toarray()).to(device)
y_train1 = torch.LongTensor(numpy.array(y_train1)).to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

# тренируем модель
num_epochs = 1000
for epoch in range(num_epochs):
    model1.train()
    optimizer.zero_grad()

    outputs = model1(X_train_tfidf)

    loss = criterion(outputs, y_train1)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# конвертируем данные в последовательность, а затем в тензор
y_test1 = torch.tensor(numpy.array(y_test1), dtype=torch.long).to(device)
X_test_tfidf_dense1 = torch.FloatTensor(X_test_tfidf1.toarray()).to(device)

model1.eval()
with torch.no_grad():
    outputs = model1(X_test_tfidf_dense1)
    _, predicted1 = torch.max(outputs, 1)

correct1 = (predicted1 == y_test1).sum().item()
total1 = y_test1.size(0)
accuracy1 = correct1 / total1 * 100
print(f'Test Accuracy: {accuracy1:.2f}%')
# оценка точности модели

w_f1_score1 = f1_score(y_test1, predicted1, average='weighted')
print(f'F1 Score: {w_f1_score1:.4f}')

# сохраним модель
torch.save(model1, 'model_LSTM1.pth')
