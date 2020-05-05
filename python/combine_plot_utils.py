import matplotlib.pyplot as plt
from datetime import date

def to_integer(string):
  '''
  If string represents an integer, return the integer, otherwise return None
  https://stackoverflow.com/q/354038
  '''
  try:
    return int(string)
  except ValueError:
    return None


def form_date (eight_digits_date):
  ''' Takes a string with eight digits, returns the corresponding date in a nicer format '''
  ## First, check input string
  if not to_integer(eight_digits_date):
    return None
  if not len(eight_digits_date) == 8:
    return None

  ## By default, assume the format YYYYMMDD
  isodate_string = f'{eight_digits_date[0:4]}-{eight_digits_date[4:6]}-{eight_digits_date[6:8]}'

  ## This check assumes the year is 202Y
  if eight_digits_date[0:3] != '202':
    ## if not 202YMMDD, try DDMM202Y
    if eight_digits_date[4:7] == '202':
      isodate_string = f'{eight_digits_date[4:8]}-{eight_digits_date[2:4]}-{eight_digits_date[0:2]}'
    ## Otherwise, give up
    else:
      return None

  ## Return date string in the desired format
  return date.fromisoformat(isodate_string).strftime('%B %d, %Y')


def form_title (meta, objname):
  ''' Form high-quality plot title from metadata '''
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
      ## For other valid test numbers, actually use default title above

  ## Now append the date, if valid
  date_string = form_date(meta[objname]['Date'])
  if date_string:
    title += " on %s" % date_string

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
