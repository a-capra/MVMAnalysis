
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
  maximum_bias_error_volume = 40         # A in ml
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


MVM = Ventilator(2, 0.04, 2, 0.04, 40, 0.15)
