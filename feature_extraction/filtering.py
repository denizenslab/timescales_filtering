"""Module for designing and applying filters."""
import scipy.signal
import scipy.fftpack
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt


class FIR:
    """Finite impulse response filter

    Parameters
    ----------
    fir : array
        Finite impulse response (FIR) filter

    Examples
    --------
    >>> f = FIR(fir=[0.2, 0.6, 0.2])
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fir=np.ones(1), fs=1.0, fir_imag=None):
        self.fir = fir
        self.fs = fs
        self.fir_imag = fir_imag

    def _transform(self, fir, sigin):
        """Apply this filter to a signal

        Parameters
        ----------
        sigin : array, shape (n_points, ) or (n_points, n_signals)
            Input signal

        Returns
        -------
        out : array, shape (n_points, ) or (n_points, n_signals)
            Filtered signal
        """
        sigin_ndim = sigin.ndim
        sigin = np.atleast_2d(sigin)

        out = np.empty(sigin.shape)

        for i, sig in enumerate(sigin.T):
            tmp = scipy.signal.fftconvolve(sig, fir, "same")
            out[:, i] = tmp

        if sigin_ndim == 1:
            out = out[0]
        else:
            out = np.asarray(out)

        return out

    def _plot(self, fir, xlim=[0, 1], axes=None, fscale="linear", label=None):
        """
        Plots the impulse response and the transfer function of the filter.
        """
        fig_passed = axes is not None
        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        else:
            axes = np.atleast_1d(axes)
            if np.any([not isinstance(ax, plt.Axes) for ax in axes]):
                raise TypeError(
                    "axes must be a list of matplotlib Axes, got {}"
                    " instead.".format(type(axes))
                )
            if len(axes) < 2:
                raise ValueError(
                    "Passed figure must have at least two axes"
                    ", given figure has {}.".format(len(axes))
                )
            fig = axes[0].figure

        fir = np.atleast_1d(fir)
        order = fir.shape[0]

        # compute periodogram
        nfft = max(int(2 ** np.ceil(np.log2(order))), 1024)
        frequencies = np.linspace(0, self.fs / 2, nfft // 2 + 1)
        spec = np.abs(np.fft.rfft(fir, n=nfft))
        axes[0].plot(frequencies, spec, label=label)
        axes[0].fill_between(frequencies, spec * 0, spec * (spec > 0.5), alpha=0.3)
        axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.5)
        axes[0].set_xscale(fscale)
        axes[0].set_xlim(xlim)
        axes[0].set_title("Transfer function of FIR filter")
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].legend()

        # plots
        time = (np.arange(order) - order // 2) / self.fs
        axes[1].plot(time, fir, label=label)
        axes[1].set_title("Impulse response of FIR filter")
        axes[1].set_xlabel("Time (sec)")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend()
        if not fig_passed:
            fig.tight_layout()
        return fig

    def transform(self, sigin):
        """Apply this filter to a signal

        Parameters
        ----------
        sigin : array, shape (n_points, ) or (n_signals, n_points)
            Input signal

        Returns
        -------
        filtered : array, shape (n_points, ) or (n_signals, n_points)
            Filtered signal
        (filtered_imag) : array, shape (n_points, ) or (n_signals, n_points)
            Only when extract_complex is true.
            Filtered signal with the imaginary part of the filter
        """
        filtered = self._transform(self.fir, sigin)

        if self.fir_imag is not None:
            filtered_imag = self._transform(self.fir_imag, sigin)
            return filtered, filtered_imag
        else:
            return filtered

    def plot(self, xlim=[0, 1], axes=None, fscale="linear"):
        """
        Plots the impulse response and the transfer function of the filter.
        """
        fig = self._plot(self.fir, xlim=xlim, axes=axes, fscale=fscale, label="real")

        if self.fir_imag is not None:
            axes = fig.axes if axes is None else axes
            self._plot(self.fir_imag, xlim=xlim, axes=axes, fscale=fscale, label="imag")

        return fig


def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency at some time.
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc
    function will be non-zero.
    """
    t = t * cutoff
    val = (
        window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi**2 * t**2)
    )
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val / (val.sum() + 1e-10)


class LanczosFilter(FIR):
    """Low-pass Lanczos filter

    Parameters
    ----------
    fs : float
        Sampling frequency
    fc : float
        Cut-off frequency of the low-pass filter
    window : float
        Number of lobes
    extract_complex : boolean, (default False)
        If True, ``transform`` returns a second signal with zeros, to mimic
        the BandPassFilter behavior.

    Examples
    --------
    >>> f = LanczosFilter(fs=20., fc=0.5)
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fs, fc, window=3.0, extract_complex=False):
        self.fs = fs
        self.fc = fc
        self.window = window
        self.extract_complex = extract_complex

        self._design()

    def _design(self):
        fir = lanczosfun(
            self.fc * 2, np.linspace(-6, 6, int(self.fs * 12) + 1), window=self.window
        )

        # the filter must be symmetric, in order to be zero-phase
        self.fir = (fir + fir[::-1]) / 2

        if self.extract_complex:
            self.fir_imag = fir * 0.0
        else:
            self.fir_imag = None

        return self


class LowPassFilter(FIR):
    """Low-pass FIR filter

    Designs a FIR filter that is a low-pass filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    fc : float
        Cut-off frequency of the low-pass filter
    bandwidth : float
        Bandwidth of the FIR wavelet filter
    ripple_db : float (default 60.0)
        Positive number specifying maximum ripple in passband (dB) and minimum
        ripple in stopband, in Kaiser-window low-pass FIR filter.
    extract_complex : boolean, (default False)
        If True, ``transform`` returns a second signal with zeros, to mimic
        the BandPassFilter behavior.

    Examples
    --------
    >>> f = LowPassFilter(fs=20., fc=3., bandwidth=3.)
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fs, fc, bandwidth, ripple_db=60.0, extract_complex=False):
        self.fs = fs
        self.fc = fc
        self.bandwidth = bandwidth
        self.ripple_db = ripple_db
        self.extract_complex = extract_complex

        self._design()

    def _design(self, filter_type="blackman"):
        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = scipy.signal.kaiserord(self.ripple_db, self.bandwidth / self.fs)

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        if filter_type == "kaiser":
            fir = scipy.signal.firwin(N, self.fc / self.fs * 2, window=("kaiser", beta))
        elif filter_type == "blackman":
            fir = scipy.signal.firwin(
                N, self.fc / self.fs * 2, pass_zero="lowpass", window="blackman"
            )

        # the filter must be symmetric, in order to be zero-phase
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        self.fir = fir / np.sum(fir)

        if self.extract_complex:
            self.fir_imag = fir * 0.0
        else:
            self.fir_imag = None

        return self


class HighPassFilter(FIR):
    """High-pass FIR filter

    Designs a FIR filter that is a low-pass filter.

    Parameters
    ----------
    fs : float
        Sampling frequency
    fc : float
        Cut-off frequency of the low-pass filter
    bandwidth : float
        Bandwidth of the FIR wavelet filter
    ripple_db : float (default 60.0)
        Positive number specifying maximum ripple in passband (dB) and minimum
        ripple in stopband, in Kaiser-window low-pass FIR filter.
    extract_complex : boolean, (default False)
        If True, ``transform`` returns a second signal with zeros, to mimic
        the BandPassFilter behavior.

    Examples
    --------
    >>> f = HighPassFilter(fs=20., fc=3., bandwidth=3.)
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fs, fc, bandwidth, ripple_db=60.0, extract_complex=False):
        self.fs = fs
        self.fc = fc
        self.bandwidth = bandwidth
        self.ripple_db = ripple_db
        self.extract_complex = extract_complex

        self._design()

    def _design(self, filter_type="blackman"):
        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = scipy.signal.kaiserord(self.ripple_db, self.bandwidth / self.fs * 2)

        N |= 1

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        if filter_type == "kaiser":
            fir = scipy.signal.firwin(
                N, self.fc / self.fs * 2, window=("kaiser", beta), pass_zero="highpass"
            )
        elif filter_type == "blackman":
            fir = scipy.signal.firwin(
                N, self.fc / self.fs * 2, pass_zero="highpass", window="blackman"
            )

        self.fir = fir / np.sum(fir)

        if self.extract_complex:
            self.fir_imag = fir * 0.0
        else:
            self.fir_imag = None

        return self


class BandPassFilter(FIR):
    """Band-pass FIR filter

    Designs a band-pass FIR filter centered on frequency fc.

    Parameters
    ----------
    fs : float
        Sampling frequency
    fc : float
        Center frequency of the bandpass filter
    n_cycles : float or None, (default 7.0)
        Number of oscillation in the wavelet. None if bandwidth is used.
    bandwidth : float or None, (default None)
        Bandwidth of the FIR wavelet filter. None if n_cycles is used.
    zero_mean : boolean, (default True)
        If True, the mean of the FIR is subtracted, i.e. fir.sum() = 0.
    extract_complex : boolean, (default False)
        If True, the wavelet filter is complex and ``transform`` returns two
        signals, filtered with the real and the imaginary part of the filter.

    Examples
    --------
    >>> f = BandPassFilter(fs=20., fc=5., bandwidth=1.)
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(
        self,
        fs,
        fc,
        bandwidth=0.5,
        n_cycles=None,
        zero_mean=True,
        extract_complex=False,
        window_type="blackman",
    ):
        self.fc = fc
        self.fs = fs
        self.n_cycles = n_cycles
        self.bandwidth = bandwidth
        self.zero_mean = zero_mean
        self.extract_complex = extract_complex
        self.window_type = window_type

        self._design()

    def _design(self):
        """Designs the FIR filter"""
        order, _ = scipy.signal.kaiserord(ripple=60, width=self.bandwidth / self.fs * 2)
        order = order // 2 * 2 + 1
        half_order = (order - 1) // 2

        if self.window_type == "blackman":
            w = np.blackman(order)
        elif self.window_type == "kaiser":
            order, beta = scipy.signal.kaiserord(
                ripple=60, width=self.bandwidth / self.fs * 2
            )
            w = np.kaiser(order, beta)
        t = np.linspace(-half_order, half_order, order)
        phase = (2.0 * np.pi * self.fc / self.fs) * t

        car = np.cos(phase)
        fir = w * car

        # the filter must be symmetric, in order to be zero-phase
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        # remove the constant component by forcing fir.sum() = 0
        if self.zero_mean:
            fir -= fir.sum() / order

        gain = np.sum(fir * car)
        self.fir = fir * (1.0 / gain)

        # add the imaginary part to have a complex wavelet
        if self.extract_complex:
            car_imag = np.sin(phase)
            fir_imag = w * car_imag
            gain_imag = np.sum(fir_imag * car_imag)
            self.fir_imag = fir_imag * (1.0 / gain_imag)
        else:
            self.fir_imag = None

        return self

    def _get_order(self):
        if self.bandwidth is None and self.n_cycles is not None:
            half_order = int(float(self.n_cycles) / self.fc * self.fs / 2)
        elif self.bandwidth is not None and self.n_cycles is None:
            half_order = int(1.65 * self.fs / self.bandwidth) // 2
        else:
            raise ValueError(
                "fir.BandPassFilter: n_cycles and bandwidth "
                "cannot be both None, or both not None. Got "
                "%s and %s"
                % (
                    self.n_cycles,
                    self.bandwidth,
                )
            )

        order = half_order * 2 + 1
        return order


def apply_filter_even_grid(
    fir,
    data,
    fc,
    extract_complex=False,
    bandwidth=None,
    plot=False,
    window_type="blackman",
    fs=1,
):
    """
    Parameters
    ----------
    fir : str
    data : array of shape (n_old_time, n_features)
    fc : float: center (if bandpass) or cutoff (if lowpass) frequency.
    bandwidth : float : width of the filter (if bandpass)
    """
    # create the filters only here since we need the intermediate sampling rate
    if fir == "lanczos":
        fir = LanczosFilter(fs=fs, fc=fc)
    elif fir == "bandpass":
        fir = BandPassFilter(
            fs=fs,
            fc=fc,
            bandwidth=bandwidth,
            extract_complex=extract_complex,
            window_type=window_type,
        )
    elif fir == "lowpass":
        fir = LowPassFilter(
            fs=fs, fc=fc, bandwidth=bandwidth, extract_complex=extract_complex
        )
    elif fir == "highpass":
        fir = HighPassFilter(
            fs=fs, fc=fc, bandwidth=bandwidth, extract_complex=extract_complex
        )
    if plot:

        def remove_periods(number_str):
            return str(number_str).replace(".", "")

        fir.plot(xlim=[fc - 2 * bandwidth, fc + 2 * bandwidth])
        plt.savefig(
            f"./figures/filters_even_grid/fc{remove_periods(fc)}_fs{fs}_bandwidth{remove_periods(bandwidth)}_complex{extract_complex}"
        )
        plt.close()
    if extract_complex:
        filtered_real, filtered_imag = fir.transform(data)
        quadrature = np.sqrt(filtered_real**2 + filtered_imag**2)
    else:
        quadrature = fir.transform(data)

    return quadrature


def apply_filter(
    fir,
    data,
    old_time,
    new_time,
    fc,
    extract_complex=False,
    xlim=[0, 1],
    inter_upsampling=40,
    bandwidth=None,
    plot=False,
    window_type="blackman",
):
    """
    Parameters
    ----------
    fir : str
    data : array of shape (n_old_time, n_features)
    old_time : array of shape (n_old_time, )
    new_time : array of shape (n_new_time, )
    fc : float: center (if bandpass) or cutoff (if lowpass) frequency.
    bandwidth : float : width of the filter (if bandpass)
    """
    final_fs = 1 / np.mean(
        np.diff(new_time)
    )  # Sampling rate in herz (assuming new_time is in seconds).

    fs = final_fs * inter_upsampling  # intermediate sampling rate
    # create the filters only here since we need the intermediate sampling rate
    if fir == "lanczos":
        fir = LanczosFilter(fs=fs, fc=fc)
    elif fir == "bandpass":
        fir = BandPassFilter(
            fs=fs,
            fc=fc,
            bandwidth=bandwidth,
            extract_complex=extract_complex,
            window_type=window_type,
        )
    elif fir == "lowpass":
        fir = LowPassFilter(
            fs=fs, fc=fc, bandwidth=bandwidth, extract_complex=extract_complex
        )
    elif fir == "highpass":
        fir = HighPassFilter(
            fs=fs, fc=fc, bandwidth=bandwidth, extract_complex=extract_complex
        )
    if plot:
        fir.plot(xlim=xlim)
    inter_time = np.linspace(
        new_time.min() - 1,
        new_time.max() + 1,
        2 + int(fs * (new_time.max() + 2 - new_time.min())),
    )

    def apply_filter_on_non_regular_grid(fir_array, data):
        def func(sampled_time):
            fir_time = (np.arange(len(fir_array)) - (len(fir_array) - 1) // 2) / fs
            return scipy.interpolate.interp1d(
                fir_time, fir_array, bounds_error=False, fill_value=0
            )(sampled_time)

        mat = np.zeros((len(inter_time), len(old_time)))
        for ndi in range(len(inter_time)):
            mat[ndi, :] = func(inter_time[ndi] - old_time)

        return np.dot(mat, data)

    # compute the filtered signal, at the intermediate sampling rate
    filtered_real = apply_filter_on_non_regular_grid(fir.fir, data)
    if fir.fir_imag is not None:
        filtered_imag = apply_filter_on_non_regular_grid(fir.fir_imag, data)
        quadrature = np.sqrt(filtered_real**2 + filtered_imag**2)
    else:
        quadrature = filtered_real

    # decimate, i.e. apply antialiasing lowpass filter, then downsample
    decimated = scipy.signal.decimate(quadrature, q=inter_upsampling, axis=0)
    # interpolate to a potentially nonregular grid
    decimated_time = inter_time[::inter_upsampling]
    result = np.array(
        [scipy.interpolate.interp1d(decimated_time, dd)(new_time) for dd in decimated.T]
    ).T

    return result
