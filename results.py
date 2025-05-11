from helper import *



def plot_single_bar_chart(categories, values, labels, output_path, y_max=20):
    plt.figure(figsize=(6, 5))
    positions = [i * 0.4 for i in range(len(categories))] 
    bars = plt.bar(positions, values, color=(0.3, 0.6, 0.8, 0.6), width=0.2)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.5, labels[i],
                 ha='center', va='bottom', fontsize=8)

    plt.xticks(positions, categories, rotation=45, ha='right')
    plt.yticks(range(1,21))
    plt.ylabel("Exact Match")
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.show()

categories = ['Q1', 'Q2','Q3','Q4','Q5','Q6','Q7','Q8']
values = [5, 9, 4,8,0,8,11,10]
labels = ['n = 5', 'n = 9', 'n = 4', 'n=8','n=0', 'n=8', 'n=11', 'n=10']
output_path = "exactmatch.png"

# plot_single_bar_chart(categories, values, labels, output_path)

def plot_grouped_bar_with_error(bars1, bars2, bars3, labels, bar_width=0.25, output_path="adddrop.png"):
    r1 = np.arange(len(labels)) 
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.figure(figsize=(20, 12))
    plt.bar(r1, bars1, width=bar_width, color='lightsteelblue', edgecolor='black', capsize=7, label='Added')
    plt.bar(r2, bars2, width=bar_width, color='lightsalmon', edgecolor='black', capsize=7, label='Removed')
    plt.bar(r3, bars3, width=bar_width, color='orchid', edgecolor='black', capsize=7, label='Exact Match')

    for r_group, values in zip([r1, r2, r3], [bars1, bars2, bars3]):
        for i, value in enumerate(values):
            if i < len(r_group):
                plt.text(r_group[i], value + 0.5, str(value), ha='center', va='bottom', fontsize=12)

    plt.xticks([r + bar_width for r in r1], labels)
    plt.ylabel('Survey Response')
    plt.yticks(range(1, 21))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="png")
    plt.show()

bars1 = [13, 7, 14, 9, 13, 7, 6, 7] # 13 7 14 added
bars2 = [11, 10, 6, 6, 12, 6, 5, 4] # 11 10 6 removed
bars3 = [5, 9, 4, 8 , 0, 8, 11, 10 ] # 5 9 4 exact match
labels = ['Q1', 'Q2','Q3','Q4','Q5','Q6','Q7','Q8']


# plot_grouped_bar_with_error(bars1, bars2,bars3, labels)



categories = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']
exact_match = [5, 9, 4, 8, 0, 8, 11, 10]
partial_match = [8, 4, 9, 5, 5, 6, 5, 5]

bar_width = 0.6
x = range(len(categories))

plt.figure(figsize=(10, 8))

# Plot exact match at bottom
plt.bar(x, exact_match, color="#6A8DBF", label='Exact Match', width=bar_width)

# Plot partial match on top of exact match
plt.bar(x, partial_match, bottom=exact_match, color='#FFA067', label='Partial Match', width=bar_width)

plt.xticks(x, categories)
plt.ylabel("Survey response")
plt.yticks(range(1, 21))

legend_elements = [
    mpatches.Patch(color='#6A8DBF', label='Exact Match'),
    mpatches.Patch(color='#FFA067', label='One code add/remove')
]
plt.legend(handles=legend_elements)


plt.ylabel("Survey response")
plt.yticks(range(1,21))
plt.tight_layout()
# plt.savefig("exactmatch.png", format="png")
plt.show()



# Sample data
x = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8']
y = [5, 3, 7, 4, 8, 9, 1, 2]
add = [13, 7,14, 9, 13, 7, 6, 7]
drop = [11, 10, 6, 6, 12, 6, 5, 4]
em = [5, 9, 4, 8 , 0, 8, 11, 10]
humanerror = [5, 1, 10, 8, 13, 6, 3, 2]
llmaerror = [10, 10, 6, 4, 7, 6, 6, 8]

# Set Seaborn style
plt.figure(figsize=(8, 6))

# # Solid line with circle markers
sns.lineplot(x=x, y=em, linestyle='-', marker='s', markersize=8, label='ExactMatch', color='purple') 

# # Dashed line with square markers
# sns.lineplot(x=x, y=add, linestyle='--', marker='s', markersize=8, label='Added', color='green') 

# # Dash-dot line with triangle up markers
# sns.lineplot(x=x, y=drop, linestyle='-.', marker='^', markersize=20, label='Removed', color='purple') 

# Dotted line with asterisk markers
sns.lineplot(x=x, y=humanerror, linestyle=':', marker='*', markersize=15, label='HumanError', color='orange') 
sns.lineplot(x=x, y=llmaerror, linestyle='--', marker='>', markersize=15, label='LlamaError', color='blue') 


plt.ylabel('Survey Response') 
plt.yticks(range(1,21))
plt.legend(loc='upper right') 
# plt.savefig('humanerrorsvsllama.png',format="png")
plt.show() 





# Data

q7 = [7, 10, 12, 11, 11, 10, 7]
q5 = [0, 1, 1, 0, 3, 3, 3]

x_labels = [3, 4, 5, 6, 7, 8, 9]
x = list(range(len(x_labels)))

# Plot settings
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

# Plot lines
plt.plot(x, q7, color='orange', marker='s', linewidth=2, label='Q7')
plt.plot(x, q5, color='red', marker='s', linewidth=2, label='Q5')

# Add value labels
for i, y in enumerate(q7):
    plt.text(x[i], y + 0.8, f'{y}', color='orange', fontsize=12, ha='center')

for i, y in enumerate(q5):
    plt.text(x[i], y + 0.8, f'{y}', color='red', fontsize=12, ha='center')



# Axis settings
plt.xticks(ticks=x, labels=x_labels, fontsize=10)
plt.yticks(range(0, 21))
plt.xlabel("Yes Count", fontsize=10)
plt.ylabel("Survey Response", fontsize=10)

# Title and legend
plt.legend(loc='upper right', fontsize=10)

# Display
plt.grid(False)
plt.tight_layout()
# plt.savefig("test.png", format="png")
plt.show()
