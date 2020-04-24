import matplotlib.pyplot as plt

def form_title (meta, objname):
  title = "Test n %s"%meta[objname]['test_name']
  return title

def set_plot_title (ax, meta, objname):
  ax.set_title(form_title(meta, objname), weight='heavy')

def set_plot_suptitle (fig, meta, objname):
  fig.suptitle(form_title(meta, objname), weight='heavy')

def save_figure (mypyplot, plottype, meta, objname, sitename, output_directory, web):
  figpath = "%s/%s_%s_%s.png" % (output_directory, meta[objname]['Campaign'], plottype, objname.replace('.txt', ''))  #TODO make sure it is correct, or will overwrite!
  if web:
      figpath = "%s/%s_%s_test%s_run%s_%s.png" % (output_directory, sitename, meta[objname]['Date'], meta[objname]['test_name'], meta[objname]['Run'], plottype)
  print(f'Saving figure to {figpath}')
  mypyplot.savefig(figpath)
