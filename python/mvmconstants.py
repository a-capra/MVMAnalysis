
class Ventilator:
  '''
  Constants about ventilators: specifications, etc.
  Initialized to MHRA values, accessed April 21, 2020
  https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/879382/RMVS001_v4.pdf
  '''

  ## Maximum errors as per requirements for certification
  maximum_bias_error_peep = 2            # A in cmH2O
  maximum_linearity_error_peep = 0.04    # B/100 for PEEP
  maximum_bias_error_pinsp = 2           # A in cmH2O
  maximum_linearity_error_pinsp = 0.04   # B/100 for Pinsp
  maximum_bias_error_volume = 4          # A in ml
  maximum_linearity_error_volume = 0.15  # B/100 for tidal volume

  ## There probably exist other ventilator constants of interest

  def __init__(self, A_peep, B_peep, A_pinsp, B_pinsp, A_vol, B_vol):
    '''
    Instantiate ventilator object by specifying maximum errors:
    Define maximum bias error A, maximum linearity error B
    EXAMPLE ±(A +(B % of the set pressure)) cmH2O
    '''
    self.maximum_bias_error_peep = A_peep          # A in cmH2O
    self.maximum_linearity_error_peep = B_peep     # B/100 for PEEP
    self.maximum_bias_error_pinsp = A_pinsp        # A in cmH2O
    self.maximum_linearity_error_pinsp = B_pinsp   # B/100 for Pinsp
    self.maximum_bias_error_volume = A_vol         # A in ml
    self.maximum_linearity_error_volume = B_vol    # B/100 for tidal volume

MVM = Ventilator(2, 0.04, 2, 0.04, 4, 0.15)


class PressureTest:
  '''
  Ventilator test with lung simulator
  '''
  TV = None        # ml
  C = None         # ml/hPa
  R = None         # hPa/l/s
  leakage = None   # ml/min
  rate = None      # breaths/min
  I = None         # s
  deltaP = None    # hPa
  PEEP = None      # hPa
  Pinsp = None     # hPa
  O2 = None        # percent

  def __init__(self, TV, C, R, leakage, rate, I, deltaP, PEEP, O2=21):
    self.TV = TV
    self.C = C
    self.R = R
    self.leakage = leakage
    self.rate = rate
    self.I = I
    self.deltaP = deltaP
    self.PEEP = PEEP
    self.Pinsp = PEEP + deltaP
    self.O2 = O2

  def __eq__(self, other):
    epsilon = 1e-3
    return (
      abs(self.TV - other.TV) < epsilon and
      abs(self.C - other.C) < epsilon and
      abs(self.R - other.R) < epsilon and
      abs(self.leakage - other.leakage) < epsilon and
      abs(self.rate - other.rate) < epsilon and
      abs(self.I - other.I) < epsilon and
      abs(self.deltaP - other.deltaP) < epsilon and
      abs(self.PEEP - other.PEEP) < epsilon
    )
      #abs(self.O2 - other.O2) < epsilon   # excluded from comparison as we don't enforce O2 fraction

  def print_comparison(self, other):
    print(f"""
      Tidal Volume  {self.TV} - {other.TV}
      Compliance    {self.C} - {other.C}
      Resistance    {self.R} - {other.R}
      Leakage       {self.leakage} - {other.leakage}
      Rate          {self.rate} - {other.rate}
      Inspiration   {self.I} - {other.I}
      Delta P       {self.deltaP} - {other.deltaP}
      PEEP          {self.PEEP} - {other.PEEP}
    """)


StandardTests = {
  '1201' : PressureTest(500, 50,   5,     0, 20, 1., 10,  5, 30),
  '1202' : PressureTest(500, 50,  20,     0, 12, 1., 15, 10, 90),
  '1203' : PressureTest(500, 20,   5,     0, 20, 1., 25,  5, 90),
  '1204' : PressureTest(500, 20,  20,     0, 20, 1., 25, 10, 30),
  '1205' : PressureTest(300, 20,  20,     0, 20, 1., 15,  5, 30),
  '1206' : PressureTest(300, 20,  50,     0, 12, 1., 25, 10, 90),
  '1207' : PressureTest(300, 10,  50,     0, 20, 1., 30,  5, 90),
  '1208' : PressureTest(200, 10,  10,     0, 20, 1., 25, 10, 30),
  '1209' : PressureTest( 50,  3,  10,     0, 30, .6, 15,  5, 30),
  '1210' : PressureTest( 50,  3,  20,     0, 30, .6, 15, 10, 30),
  '1211' : PressureTest( 50,  3,  50,     0, 20, .6, 25,  5, 60),
  '1212' : PressureTest( 30,  3,  20,     0, 30, .6, 10,  5, 30),
  '1213' : PressureTest( 30,  3,  50,     0, 20, .6, 15, 10, 90),
  '1214' : PressureTest( 30,  1,  20,     0, 30, .6, 30,  5, 90),
  '1215' : PressureTest( 30,  1, 100,     0, 30, .6, 30, 10, 30),
  '1216' : PressureTest( 20,  1, 200,     0, 50, .4, 20,  5, 30),
  '1217' : PressureTest( 15,  1, 200,     0, 50, .4, 15, 10, 60),
  '1218' : PressureTest( 10,  1,  50,     0, 60, .4, 10,  5, 60),
  '1219' : PressureTest(  5, .5,  50,     0, 60, .4, 15, 10, 60),
  '1220' : PressureTest(  5, .5,  50,     0, 30, .4, 10,  5, 30),
  '1221' : PressureTest(  5, .5, 200,     0, 60, .4, 15, 10, 30),
  '8001' : PressureTest(500, 50,   5,     0, 20, 1., 10,  5),
  '8002' : PressureTest(500, 50,  20,     0, 12, 1., 15, 10),
  '8003' : PressureTest(500, 20,   5,     0, 20, 1., 25,  5),
  '8004' : PressureTest(500, 20,  20,     0, 20, 1., 25, 10),
  '8005' : PressureTest(500, 50,   5,  5000, 20, 1., 25,  5),
  '8006' : PressureTest(500, 50,  20, 10000, 12, 1., 25, 10),
  '8007' : PressureTest(300, 20,  20,     0, 20, 1., 15,  5),
  '8008' : PressureTest(300, 20,  50,     0, 12, 1., 25, 10),
  '8009' : PressureTest(300, 10,  50,     0, 20, 1., 30,  5),
  '8010' : PressureTest(300, 20,  20,  3000, 20, 1., 25,  5),
  '8011' : PressureTest(300, 20,  50,  6000, 12, 1., 25, 10),
  '8012' : PressureTest(200, 10,  20,     0, 20, 1., 25, 10),
}
