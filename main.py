import numpy as np
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
from dataset import preprocess_data, test

ALPHABET = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
    6: "g",
    7: "h",
    8: "i",
    9: "j",
    10: "k",
    11: "l",
    12: "m",
    13: "n",
    14: "o",
    15: "p",
    16: "q",
    17: "r",
    18: "s",
    19: "t",
    20: "u",
    21: "v",
    22: "w",
    23: "x",
    24: "y",
    25: "z"
}


def activate(x):
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.weights_input_to_hidden = np.random.normal(0.0, pow(self.input_neurons, -0.5),
                                                        (self.hidden_neurons, self.input_neurons))
        self.weights_hidden_to_output = np.random.normal(0.0, pow(self.hidden_neurons, -0.5),
                                                         (self.output_neurons, self.hidden_neurons))

    def train(self, train_data, target_data):
        train_data = np.array(train_data, ndmin=2).T
        target_data = np.array(target_data, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden, train_data)
        hidden_outputs = activate(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = activate(final_inputs)

        output_errors = target_data - final_outputs
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)

        self.weights_hidden_to_output += 0.1 * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                np.transpose(hidden_outputs))
        self.weights_input_to_hidden += 0.1 * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                               np.transpose(train_data))

    def query(self, inputs_list):
        query_inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden, query_inputs)
        hidden_outputs = activate(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = activate(final_inputs)

        return final_outputs


print('Начало работы\n')
training_data, target = preprocess_data('latin_alphabet/', 260, 1024)
print('Входные данные: ', training_data, '\n')
print('Выходные данные: ', target, '\n')
network = Network(1024, 512, 26)

print("\nПроверка нейронной сети до тренировки на букве 'a'\n")
result = network.query(training_data[2])
for j in range(len(result)):
    print(ALPHABET.get(j) + ") ", result[j])
print("\n")

for e in range(100):
    print('Эпоха: ', e + 1)
    for i in range(len(training_data)):
        inputs = training_data[i]
        targets = np.zeros(26) + 0.01
        targets[int(target[i])] = 0.99
        network.train(inputs, targets)

print("\nПроверка нейронной сети после тренировки на букве 'c'\n")
result = network.query(training_data[2])
for j in range(len(result)):
    print(ALPHABET.get(j) + ") ", result[j])


def openfile():
    filepath = filedialog.askopenfilename(initialdir="test",
                                          title="Тест",
                                          filetypes=(("png файлы", "*.png"),
                                                     ("png файлы", "*.png")))
    test_image = test(filepath, 1024)
    imgt = ImageTk.PhotoImage(Image.open(filepath))
    panel.config(image=imgt)
    panel.image = imgt

    number_of_answer = ""
    max_weight = 0
    print("Нейросеть думает...\n")
    test_data = network.query(test_image)
    for g in range(len(test_data)):
        print(ALPHABET.get(g) + ") ", test_data[g])
        if max_weight < test_data[g]:
            max_weight = test_data[g]
            number_of_answer = g
    answer = ALPHABET.get(number_of_answer)

    label2.config(text=answer)
    print("\nНейросеть думает, что на картинке: ", answer, '\n')


window = Tk()
window.title("Хисамов Искандер Лабораторная работа №2.1")
window.geometry("550x350+700+400")
button = Button(text="Загрузить картинку", command=openfile)
button.pack(fill=BOTH, expand=0)
frame = Frame(window, relief=RAISED, borderwidth=1)
frame.pack(fill=BOTH, expand=True)
img = ImageTk.PhotoImage(Image.open("test/00.png"))
panel = Label(frame, image=img)
panel.pack(side="bottom", fill="both", expand="yes")
label1 = Label(text="На картинке: ")
label1.pack(side=LEFT, padx=5, pady=5)
label2 = Label(text="a")
label2.pack(side=LEFT)
window.mainloop()


