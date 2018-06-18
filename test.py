import matplotlib.pyplot as plt

fig= plt.figure()
plt.plot(range(10))
fig.savefig("save_file_name.png")
plt.close()
