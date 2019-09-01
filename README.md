# Data Science Tools

## Usage

```
pip install --upgrade git+git://github.com/edwmurph/ds.git
```

Load module in python script:
```
from tools import load_data

df = load_data.laod_python_data(..)
```

## Cheat sheet

TODO find a better spot for these

Create figure with multiple subplots:
```
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
plt.subplots_adjust(hspace=.3, wspace=.2)
```
Graph line:
```
f = lambda x: np.sqrt(x)
X = np.linspace(0, 10)
Y = f(X)
ax1.plot(X, Y, label='sqrt fn', lw=1, ls='--', color='red')
```
