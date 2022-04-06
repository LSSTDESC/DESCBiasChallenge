
import pyccl.nl_pt as pt

class PTMatterTracer(pt.PTTracer):
    """:class:`PTTracer` representing matter fluctuations.
    """
    def __init__(self):
        self.biases = {}
        self.type = 'M'


class PTNumberCountsTracer(pt.PTTracer):
    """:class:`PTTracer` representing number count fluctuations.
    This is described by 1st and 2nd-order biases and
    a tidal field bias. These are provided as floating
    point numbers or tuples of (reshift,bias) arrays.
    If a number is provided, a constant bias is assumed.
    If `None`, a bias of zero is assumed.
    Args:
        b1 (float or tuple of arrays): a single number or a
            tuple of arrays (z, b(z)) giving the first-order
            bias.
        b2 (float or tuple of arrays): as above for the
            second-order bias.
        bs (float or tuple of arrays): as above for the
            tidal bias.
        b3nl (float or tuple of arrays): as above for the
            third-order bias.
        bk2 (float or tuple of arrays): as above for the
            non-local bias.
        sn (float or tuple of arrays): as above for the
            residual shot-noise.
    """
    def __init__(self, b1, b2=None, bs=None, b3nl=None, bk2=None, sn=None):
        self.biases = {}
        self.type = 'NC'

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        # Initialize b2
        self.biases['b2'] = self._get_bias_function(b2)
        # Initialize bs
        self.biases['bs'] = self._get_bias_function(bs)
        # Initialize b3nl
        self.biases['b3nl'] = self._get_bias_function(b3nl)
        # Initialize bk2
        self.biases['bk2'] = self._get_bias_function(bk2)
        # Initialize sn
        self.biases['sn'] = self._get_bias_function(sn)

    @property
    def b1(self):
        """Internal first-order bias function.
        """
        return self.biases['b1']

    @property
    def b2(self):
        """Internal second-order bias function.
        """
        return self.biases['b2']

    @property
    def bs(self):
        """Internal tidal bias function.
        """
        return self.biases['bs']

    @property
    def b3nl(self):
        """Internal third-order bias function.
        """
        return self.biases['b3nl']

    @property
    def bk2(self):
        """Internal non-local bias function.
        """
        return self.biases['bk2']
    @property
    def sn(self):
        """Internal residual shot-noise function.
        """
        return self.biases['sn']


class HZPTNumberCountsTracer(pt.PTTracer):
    """:class:`PTTracer` representing number count fluctuations.
    These are provided as floating
    point numbers or tuples of (reshift,bias) arrays.
    If a number is provided, a constant bias is assumed.
    If `None`, a bias of zero is assumed.

    Args:
        b1 (float or tuple of arrays): a single number or a
            tuple of arrays (z, b(z)) giving the first-order
            bias.
        sngg (float or tuple of arrays): as above for the
            tracer shot noise.
        A0gg (float or tuple of arrays): as above for the
            HZPT gg BB amplitude.
        Rgg (float or tuple of arrays): as above for the
            HZPT gg BB compensation.
        R1hgg (float or tuple of arrays): as above for the
            first HZPT gg BB profile parameter.
        A0gm (float or tuple of arrays): as above for the
            HZPT gm BB amplitude.
        Rgm (float or tuple of arrays): as above for the
            HZPT gm BB compensation.
        R1hgm (float or tuple of arrays): as above for the
            first HZPT gm BB profile parameter.
        R1gm (float or tuple of arrays): as above for the
            second HZPT gm BB profile parameter.
        R12gm (float or tuple of arrays): as above for the
            third HZPT gm BB profile parameter.
    """
    def __init__(self, b1, sngg=None,A0gg=None,Rgg=None,R1hgg=None,
                           A0gm=None,Rgm=None,R1hgm=None,R1gm=None,R12gm=None
                           ):
        self.biases = {}
        self.type = 'NC'

        # Initialize b1
        self.biases['b1'] = self._get_bias_function(b1)
        self.biases['sngg'] = self._get_bias_function(sngg)
        self.biases['A0gg'] = self._get_bias_function(A0gg)
        self.biases['Rgg'] = self._get_bias_function(Rgg)
        self.biases['R1hgg'] = self._get_bias_function(R1hgg)
        self.biases['A0gm'] = self._get_bias_function(A0gm)
        self.biases['Rgm'] = self._get_bias_function(Rgm)
        self.biases['R1hgm'] = self._get_bias_function(R1hgm)
        #FIXME: fix scaling relation
        self.biases['R1gm'] = self._get_bias_function(R1gm)
        self.biases['R12gm'] = self._get_bias_function(R12gm)

    @property
    def b1(self):
        return self.biases['b1']

    @property
    def sngg(self):
        return self.biases['sngg']
    @property
    def A0gg(self):
        return self.biases['A0gg']
    @property
    def Rgg(self):
        return self.biases['Rgg']
    @property
    def R1hgg(self):
        return self.biases['R1hgg']

    @property
    def A0gm(self):
        return self.biases['A0gm']
    @property
    def Rgm(self):
        return self.biases['Rgm']
    @property
    def R1hgm(self):
        return self.biases['R1hgm']
    @property
    def R1gm(self):
        return self.biases['R1gm']
    @property
    def R12gm(self):
        return self.biases['R12gm']

