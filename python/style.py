import matplotlib.pyplot as plt

figwidth = 18
plt.rcParams["figure.figsize"]           =   [figwidth, figwidth*1105/1920]    # figure size in inches
plt.rcParams["lines.linewidth"]          =   3     # line width in points
plt.rcParams["font.family"]              =   "monospace"
plt.rcParams["font.monospace"]           =   "Courier New"
plt.rcParams["font.style"]               =   "normal"
plt.rcParams["font.weight"]              =   "heavy"
plt.rcParams["font.size"]                =   20.0
plt.rcParams["axes.labelweight"]         =   "bold"    # weight of the x and y labels
plt.rcParams["axes.spines.right"]        =   True
plt.rcParams["axes.formatter.useoffset"] =   True    # If True, the tick label formatter
plt.rcParams["xtick.major.size"]         =   10      # major tick size in points
plt.rcParams["xtick.minor.size"]         =   4      # minor tick size in points
plt.rcParams["xtick.direction"]          =   "in"     # direction in, out, or inout
plt.rcParams["xtick.minor.visible"]      =   True
plt.rcParams["ytick.major.size"]         =   10      # major tick size in points
plt.rcParams["ytick.minor.size"]         =   4      # minor tick size in points
plt.rcParams["ytick.direction"]          =   "in"     # direction in, out, or inout
plt.rcParams["ytick.minor.visible"]      =   True
plt.rcParams["legend.fontsize"]          =   20
plt.rcParams["legend.labelspacing"]      =   0.7    # the vertical space between the legend entries in fraction of fontsize
plt.rcParams["legend.shadow"]            =   False
plt.rcParams["legend.frameon"]           =   False   # whether or not to draw a frame around legend
plt.rcParams['axes.unicode_minus']       =   False   # fix glyph error by using normal hyphens
