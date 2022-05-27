import numpy as np


class ShowerParameterization:
    r"""Class calculating particle content per shower for
    GH and Greissen parametrizations.
    """

    def __init__(self, table_decay_e, event_tag, decay_tag):

        # pythia related params
        self.table_decay_e = table_decay_e
        self.event_tag = event_tag
        self.decay_tag = decay_tag

    def greisen(self, shower_end=2000, grammage=1, table_energy=100e15):
        # slant depths
        x = np.linspace(1, shower_end, int(shower_end / grammage))
        # x0 is the radiation length in air ~ 37 g/cm^2
        # (pdg.lbl.gov/2015/AtomicNuclearProperties/HTML/air_dry_1_atm.html
        x_naught = 36.62
        # critical energy for electromagnetic cascades in the air
        e_c = 87.92e6
        # energy of photon initiated shower (eV) Gaisser 303
        e_0 = self.table_decay_e * table_energy
        beta_0 = np.log(e_0 / e_c)
        t = x / x_naught

        # age parameter, n = 0
        s = (0 + 3 * t) / (t + 2 * beta_0)

        x_max = x_naught * np.log(e_0 / e_c)
        n_max = (0.31 / pow(beta_0, 1 / 2)) * (e_0 / e_c)

        term1 = 0.31 / pow(beta_0, 1 / 2)
        term2 = np.exp(t - (t * (3 / 2) * np.log(s)))

        content = term1 * term2  # particle content

        return x, content, self.event_tag

    def gaisser_hillas(
        self,
        n_max,
        x_max,
        x_0,
        p1,
        p2,
        p3,
        shower_start: int = 0,
        shower_end: int = 2000,
        grammage: int = 1,
        pad_bins_with: float = np.nan,
        pad_tails_with: float = 0,
        fit_break_thresh: float = 1e10,
    ):

        padded_vec_len = (shower_end / grammage) + 400
        scaled_n_max = n_max * self.table_decay_e

        # allows negative starting depths
        x = np.arange(np.round(x_0), shower_end + 1, grammage)  # slant depths g/cm^2
        # print(np.round(x_0))
        # calculating gaisser-hillas function
        gh_lambda = p1 + p2 * x + p3 * (x ** 2)

        exp1 = (x_max - x_0) / gh_lambda

        term1 = np.array(
            scaled_n_max * np.nan_to_num(((x - x_0) / (x_max - x_0)) ** exp1),
            dtype=np.float64,
        )

        exp2 = (x_max - x) / gh_lambda
        term2 = np.exp(exp2)

        f = np.nan_to_num(term1 * term2)
        f = np.round(f, 0)

        # constrain the showers physically

        # pads with 0s at the end if the fit brakes and there's a pole
        if np.min(f) < 0:
            nose_dive = int(np.argwhere(f < 0)[0])
            f[nose_dive:] = pad_tails_with
        # we consider the fit broken if it exceedes a reasonable value
        if np.max(f) > fit_break_thresh:

            pole = int(np.argwhere(f > fit_break_thresh)[0])
            f[pole:] = pad_tails_with

        # floor the partilce content if < 1
        f[((f > 0) & (f < 1))] = 0

        # cut showers that are decresearing before x_0
        next_element_minus_previous = np.diff(f)
        # flat showers, meaing they contribute nothing, always 0
        if np.max(next_element_minus_previous[1:]) == 0:
            f = np.zeros(10)
        else:
            # get the index where the shower starts to increase intially,
            # pad with 0s up to there
            physical_start_point = int(
                np.argwhere(next_element_minus_previous[1:] > 0)[0]
            )
            f[:physical_start_point] = 0

        # correct zombie showers
        # i.e. showers that stay 0 for a while after nmax
        # but then starts to rebound >10000g/cm^2 after death
        nmax_idx = int(np.argmax(f[2:])) + 2
        num_of_zeroes = np.count_nonzero(f[nmax_idx:] == 0)
        percent_of_zeroes = (num_of_zeroes / len(f[nmax_idx:])) * 100
        # if after the nmax, % of the entries are 0 and still rebound,
        # treat that as zombie
        if percent_of_zeroes > 40 and f[-1] > 0:
            stop_point = int(np.argwhere(f == 0)[-1])
            f[stop_point:] = pad_tails_with

        # LambdaAtx_max = p1 + p2*x_max + p3*(x_max**2)
        # x = (x - x_max)/36.62 #shower stage
        x = np.pad(
            x,
            (int(padded_vec_len - len(x)), 0),
            "constant",
            constant_values=pad_bins_with,
        )
        f = np.pad(f, (int(padded_vec_len - len(f)), 0), "constant", constant_values=0)
        # tag the outputs with the event number
        x = np.r_[self.event_tag, self.decay_tag, x]
        f = np.r_[self.event_tag, self.decay_tag, f]

        return x, f


# =============================================================================
#
#     def gaisser_hillas(
#         self, n_max, x_max, x_0, p1, p2, p3, shower_end=2000, grammage=1
#     ):
#
#         scaled_n_max = n_max * self.table_decay_e
#
#         # constrains starting depths
#         x = np.linspace(1, shower_end, shower_end)  # slant depths g/cm^2
#
#         # calculating gaisser-hillas function
#         print(p1.shape, p2.shape, x.shape, p3.shape)
#         v2 = np.multiply.outer(p2, x)
#         v3 = np.multiply.outer(p3, x ** 2)
#         gh_lambda = p1[:, None] + v2 + v3
#         print(gh_lambda.shape)
#
#         exp1 = (x_max - x_0)[:, None] / gh_lambda
#
#         a1 = x - x_0[:, None]
#         a2 = x_max - x_0
#         a3 = a1 / a2[:, None]
#         a4 = a3 ** exp1
#         term1 = scaled_n_max[:, None] * np.nan_to_num(a4)
#
#         exp2 = (x_max[:, None] - x) / gh_lambda
#         term2 = np.exp(exp2)
#
#         f = np.nan_to_num(term1 * term2)
#
#         return x, f, self.event_tag
#
#     def single_particle_showers(self, gh_params = sefelectron_gh, tau_energies):
#         shower = ShowerParameterization(
#             table_decay_e=tau_energies, event_tag=tau_energies
#         )
#         depths, _, _ = shower.gaisser_hillas(
#             n_max=gh_params[:, 4],
#             x_max=gh_params[:, 5],
#             x_0=gh_params[:, 6],
#             p1=gh_params[:, 7],
#             p2=gh_params[:, 8],
#             p3=gh_params[:, 9],
#         )
#         return depths
# =============================================================================
