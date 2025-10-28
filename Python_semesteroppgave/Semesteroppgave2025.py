import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

# Funksjoner for farger, størrelse og label
def index_from_nedbor(x):
    if x < 1300: return 0
    if x < 1700: return 1
    if x < 2500: return 2
    if x < 3200: return 3
    return 4

def color_from_nedbor(nedbor):
    return colors[index_from_nedbor(nedbor)]

def size_from_nedbor(nedbor):
    return 350

def label_from_nedbor(nedbor):
    return str(int(nedbor / 100))

# Tegne kartet
def draw_the_map():
    axMap.cla()
    axMap.imshow(img, extent=(0, 13, 0, 10))
    df_year = df.groupby(['X', 'Y']).agg({'Nedbor': 'sum'}).reset_index()
    xr = df_year['X'].tolist()
    yr = df_year['Y'].tolist()
    nedborAar = df_year['Nedbor']
    ColorList = [color_from_nedbor(n) for n in nedborAar]
    axMap.scatter(xr, yr, c=ColorList, s=size_from_nedbor(nedborAar / 12), alpha=1)
    labels = [label_from_nedbor(n) for n in nedborAar]
    for i, y in enumerate(xr):
        axMap.text(xr[i], yr[i], s=labels[i], color='white', fontsize=10, ha='center', va='center')
    axMap.set_title("Årsnedbør Stor Bergen")
    axMap.axis('off')

# Tegner labels på x-aksen for månedene
def draw_label_and_ticks():
    xlabels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    axGraph.set_xticks(np.linspace(1, 12, 12))
    axGraph.set_xticklabels(xlabels)

# Tegner legend-boksen som forklarer fargene
def draw_legend_box():
    axLegend.axis('off')
    axLegend.set_title("Årsnedbør (mm)", fontsize=11, pad=10)
    legend_colors = ['orange', 'gray', 'blue', 'darkblue', 'black']
    legend_labels = ['<1300', '1300–1700', '1700–2500', '2500–3200', '>3200']
    for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
        y_pos = 0.9 - i * 0.18
        axLegend.add_patch(mpatches.Rectangle((0, y_pos - 0.05), 0.3, 0.1, color=color))
        axLegend.text(0.4, y_pos, f"{label} mm", fontsize=9, va='center')

# Funksjon for klikk på kartet
def on_click(event):
    global marked_point
    if event.inaxes != axMap:
        return

    marked_point = (event.xdata, event.ydata)
    x, y = marked_point

    months = np.linspace(1, 12, 12)
    vectors = np.array([[x, y, m] for m in months])

    AtPoint_df = pd.DataFrame(vectors, columns=['X', 'Y', 'Month'])

    # Gjør data om til polynomformat og predicte data
    AtPointM = poly.transform(AtPoint_df)
    y_pred = model.predict(AtPointM)
    aarsnedbor = sum(y_pred)
    gjennomsnitt = np.mean(y_pred)  # finner gjennomsnitt

    # Oppdaterer graf og kartet
    axGraph.cla()
    draw_the_map()
    axMap.set_title(f"C: ({x:.1f},{y:.1f}) - klikkpunkt (rød sirkel er estimert)")
    axMap.text(x, y, s=label_from_nedbor(aarsnedbor),
               color='white', fontsize=10, ha='center', va='center')
    axGraph.set_title(f"Nedbør per måned, Årsnedbør {int(aarsnedbor)} mm")

    colorsPred = [color_from_nedbor(nedbor * 12) for nedbor in y_pred]
    axMap.scatter(x, y, c=color_from_nedbor(aarsnedbor),
                  s=size_from_nedbor(aarsnedbor) * 3.5, marker="o")
    axMap.scatter(x, y, c="red",
                  s=size_from_nedbor(aarsnedbor) * 2.5, marker="o")

    # Tegner stolpediagram
    axGraph.bar(months, y_pred, color=colorsPred)
    draw_label_and_ticks()

    # Tegner rød linje for gjennomsnittlig månedlig nedbør i diagrammet
    axGraph.axhline(y=gjennomsnitt, color='red', linestyle='--', linewidth=2,
                    label=f"Gjennomsnitt ({gjennomsnitt:.0f} mm)")

    axGraph.grid(True, linestyle=':', alpha=0.6)

    # “Gjennomsnitt (xxx mm)"
    axGraph.legend(loc='upper right')

    plt.draw()

# Opprettet figur med 3 akser: diagram, kart og fargeforklaring
fig = plt.figure(figsize=(14, 6))

# Venstre - diagram
axGraph = fig.add_axes((0.05, 0.07, 0.30, 0.85))

# Midten - kart
axMap = fig.add_axes((0.40, 0.07, 0.45, 0.85))

# Høyre - Fargeforklaring / legend-boks
axLegend = fig.add_axes((0.87, 0.25, 0.10, 0.50))

draw_label_and_ticks()
img = mpimg.imread('StorBergen2.png')
axMap.set_title("Årsnedbør Stor Bergen")
axGraph.set_title("Per måned")
axMap.axis('off')

# Leser CSV og trener modellen
df = pd.read_csv('NedborX.csv')
marked_point = (0, 0)

X = df[['X', 'Y', 'Month']]
y = df['Nedbor']

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluerer modellen
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"R²-verdi: {r2:.2f}")
print(f"Standardavvik: {mae:.2f} mm")

colors = ['orange', 'gray', 'blue', 'darkblue', 'black']

# Tegner kart + fargeforklaring
draw_the_map()
draw_legend_box()

plt.connect('button_press_event', on_click)
plt.show()
