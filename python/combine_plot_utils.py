import matplotlib.pyplot as plt

def to_integer(string):
  '''
  If string represents an integer, return the integer, otherwise return None
  https://stackoverflow.com/q/354038
  '''
  try:
    return int(string)
  except ValueError:
    return None

def form_title (meta, objname):
  ## Default title uses test_name, which can be string
  test_name = meta[objname]['test_name']
  title = "%s Test %s" % (meta[objname]['SiteName'], meta[objname]['test_name'])

  ## First, test whether test_name is a 4-digit number
  test_number = to_integer(test_name)
  if test_number:
    if test_number >= 1000 and test_number <= 9999:
      ## Check whether we have a test from the ISO standards,
      ##   by looking at the first two digits
      ## In such a case, the last two digits are the ISO test number
      if test_number // 100 == 12:
        title = "%s ISO 80601-2-12 Test %i" % (meta[objname]['SiteName'], test_number % 100)
      elif test_number // 100 == 80:
        title = "%s ISO 80601-2-80 Test %i" % (meta[objname]['SiteName'], test_number % 100)
    ## For other valid test numbers, actually return the default title
  return title

def set_plot_title (ax, meta, objname):
  ax.set_title(form_title(meta, objname), weight='heavy')

def set_plot_suptitle (fig, meta, objname):
  fig.suptitle(form_title(meta, objname), weight='heavy')

def save_figure (mypyplot, plottype, meta, objname, output_directory, figure_format, web):
  figpath = "%s/%s_%s_%s.%s" % (output_directory, meta[objname]['Campaign'], plottype, objname.replace('.txt', ''), figure_format)  #TODO make sure it is correct, or will overwrite!
  if web:
    figpath = "%s/%s_%s_test%s_run%s_%s.%s" % (output_directory, meta[objname]['SiteName'], meta[objname]['Date'], meta[objname]['test_name'], meta[objname]['Run'], plottype, figure_format)
  print(f'Saving figure to {figpath}')
  mypyplot.savefig(figpath)
