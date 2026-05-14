# NariaiBuilder

A cosmological simulation of the Universe accounting for:
- **Lambda CDM model** (Dark Energy + Cold Dark Matter)
- Cosmological horizons and the Nariai limit
- Matter points, a central black hole, and laser feeding after the laser is activated

## Project Goal

The main objective of the simulation is to **investigate the possibility of creating a Nariai black hole** (a black hole of the maximum possible mass in an accelerating expanding universe).

The simulation is designed to find out:
- At what laser energies (using the "photon rocket" model) and at what initial time it is fundamentally possible to create a black hole at the Nariai limit.
- What maximum black hole mass can generally be created this way, overcoming cosmological expansion.

The logic of the simulation is based on building a black hole by emitting photons from surrounding matter points toward the center. These photons reach the center and increase the mass of the central black hole (CBH). In this project, it is assumed that such focused laser feeding is the most optimal way to build a supermassive black hole (up to the Nariai limit).

## Features

- Expansion of a flat ΛCDM universe according to the Friedmann equation
- Accounts for Dark Energy (Λ), Dark Matter, and Baryonic Matter
- **2D visualization of matter points** projected onto the screen
- **Info Panel**: current cosmological parameters, CMB temperature, and masses inside horizons
- **Cosmological Horizons**:
  - LTB Hubble / outer apparent horizon from the full `M(<r)` profile
  - LTB Event horizon as a radial null separatrix in the same LTB geometry
  - de Sitter Horizon (`r_dS = c/√(Λ/3)`) as the empty-Λ reference scale
  - Particle Horizon (maximum distance light has traveled)
- LTB apparent horizons for the central black hole and the cosmological outer boundary
- Laser feeding of the central black hole with discrete photon packets
- Precomputed horizon and scale-factor caches to speed up the simulation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Before the first run (or after changing `MAX_TIME_YEARS` in
`config.py`), you must precompute the cosmological horizons:

```bash
python scripts/precompute_horizons.py
```

Then run the main simulation:

```bash
python simulator.py
```

To automatically find the minimum required laser specific power (W/kg) to build a Nariai black hole using headless binary search, run:

```bash
python scripts/find_nariai_threshold.py
```

To plot a graph showing how the final Black Hole mass depends on the laser power for a specific cosmological epoch (e.g. 13.8 billion years), run:

```bash
python scripts/plot_mass_vs_power.py
```

You can specify a different cosmological epoch (time in billion years):
```bash
python scripts/plot_mass_vs_power.py --time 8.5
```

## Controls

- **Space** or **P** - pause/resume
- **←/→** - rewind/forward time while paused
- **ESC** or close window - exit

## Display Coordinate System

In `config.py`, there is a `COORDINATE_DISPLAY_MODE` parameter with string values:

- `"physical"` (default) — proper coordinates: points diverge along with the expansion of the Universe, the physical ruler has a constant length (10 billion light-years = `RULER_LENGTH_PX` px), and the comoving ruler scales as `a(t)`.
- `"comoving"` — comoving coordinates: points are at fixed χ-positions, all horizons and photons are converted to comoving by dividing by `a(t)`. The comoving ruler has a constant length, while the physical ruler shows how many comoving meters make up 10 Gly proper, meaning its length is proportional to `1/a(t)` (longer in the early universe, shorter in the far future). Physics and numerical values in the info panel do not change — only the display changes.

## Initial Matter Distribution

In `config.py`, the `MATTER_INITIAL_DISTRIBUTION` parameter sets the starting geometry of dust points (in comoving coordinates; detailed logic is in `physics/matter_simulation.py`):

- `"uniform"` (default) — uniform volume distribution projected onto the screen plane.
- `"spiral"` — projected as a spiral. It mathematically reflects a uniform distribution of matter by distance from the center, but is visualized as a uniform spiral for better structural clarity.

Also in `config.py` you can control the randomness of the generated distribution using the `MATTER_SEED` parameter:
- `MATTER_SEED = 42` (or any other integer) — uses a fixed seed to generate the exact same particle positions every time. This guarantees fully deterministic results for threshold finding.
- `MATTER_SEED = 0` — generates a random distribution on every run. Note that with 1000 particles, exact physics results (e.g., the threshold power required to reach the Nariai limit) can fluctuate by ~10% due to statistical variation in starting coordinates.

## Project Structure

- `simulator.py` - main launch file
- `config.py` - simulation parameters (time, matter, laser, etc.)
- `physics/` - physical models
  - `objects.py` - `Universe` class (cosmological simulation time)
  - `cosmology.py` - Lambda CDM parameters (Universe expansion, dark energy)
  - `apparent_horizon.py` - LTB apparent horizons and LTB event/null horizon helpers
  - `matter_simulation.py` - generation and management of matter points
  - `matter_points.py` - dynamics of matter points, laser emitters, and photons
  - `mass_calculator.py` - calculation of masses inside horizons and the CBH
  - `nariai.py` - reference vacuum SdS/Nariai calculations
- `visualization/` - visualization
  - `ui.py` - colors, labels, and UI geometry (window, rulers, horizons)
  - `renderer.py` - rendering with pygame
  - `horizons_renderer.py` - drawing cosmological horizons
  - `info_panel.py` - info panel and mass panel
  - `input_handler.py` - handling keyboard and window events
- `utils/` - utilities
  - `constants.py` - physical constants (G, c, cosmological parameters)
  - `config_utils.py` - derived values from configuration
  - `cosmology_utils.py` - auxiliary cosmological calculations
  - `format_utils.py` - formatting of physical quantities
- `data/` - precomputed horizon caches
- `scripts/` - utility scripts
  - `precompute_horizons.py` - horizon precomputation
  - `bench_calculate_masses.py` - mass calculation benchmark
  - `find_nariai_threshold.py` - headless script to find the specific power threshold using binary search
  - `plot_mass_vs_power.py` - headless script to plot BH mass vs laser power for a specific epoch
- `tests/` - tests for cosmological calculations and matter dynamics

## Physical Models

### ΛCDM and Expansion
- Flat ΛCDM model with matter and dark energy
- Planck 2018 + BAO parameters: H₀ = 67.4 km/s/Mpc, Ω_Λ = 0.685, Ω_DM = 0.266, Ω_B = 0.049
- The live simulation uses the precomputed `scale_factors` table from `data/event_horizon_cache.json` for fast `a(t)` lookup. This table is generated by `scripts/precompute_horizons.py` from the same Friedmann-equation solver used by `calculate_scale_factor_at_time`.
- If the precomputed table is missing or the requested time is outside its range, `a(t)` falls back to direct numerical Friedmann integration.
- `MAX_TIME_YEARS` in `config.py` is the single time scale (years): end of live simulation and precomputed horizon grid, future window for LTB/FLRW event-horizon integrals, and maximum post-laser wait in headless threshold/sweep scripts.
- `H(t)` and precomputed FLRW horizon tables remain available as background/reference helpers
- Scenario horizons are calculated from the current LTB state rather than by cherry-picking FLRW/SdS radii

### Matter Points and Laser
- **Strictly radial model**: matter is represented by discrete points moving strictly toward or away from the center (no angular momentum or orbits).
- **Laser emitter selection**: `EMISSION_BOUNDARY = "event"` selects points inside the LTB Event horizon; `"hubble"` selects points inside the LTB Hubble / outer apparent horizon; `"desitter"` uses the empty-universe de Sitter reference radius (`c/H_Λ = √(3/Λ)`). The `event` and `hubble` boundaries are computed from the same LTB mass profile, not from the FLRW event horizon helper.
- **Simplified fast mass mode**: `SIMPLIFIED_MASSES_BH_ONLY = True` switches horizon/mass accounting to a fast analytic approximation that ignores **discrete matter shells and in-flight laser photons** (their contribution to `M(<r)`), but still accounts for the **smooth FRW matter background** via an effective cosmological constant `Λ_eff(t) = 3·(H(t)/c)²`. In this mode:
  - **Hubble / outer apparent** is solved analytically as the outer Schwarzschild–de Sitter root with `Λ_eff(t)` — it grows with time as `H(t)` decreases and shrinks as `M_BH` grows (toward `r_N` in the Nariai limit).
  - **Inner apparent (CBH)** is the inner SdS root with the same `Λ_eff(t)` (so the displayed BH horizon includes the Λ correction, not pure Schwarzschild).
  - **Event horizon** uses the FLRW event-horizon interpolator (no `M_BH` dependence in the integrand) and is rescaled by the same shrink factor as Hubble: `r_event(t, M_BH) ≈ r_event_FLRW(t) · (r_hubble_with_BH / (c/H(t)))`. Reduces to the FLRW value at `M_BH = 0`, converges to `r_N` together with Hubble at `M_BH → M_N`.
  - Mass inside any horizon is reported as just `M_BH` (matter/laser contribution skipped).
  - Significantly reduces `ltb_horizons_masses` time at the cost of dropping the discrete-shell contribution to `M(<r)`.
- **Hubble drag**: the peculiar momentum of points scales as `p_pec ∝ 1/a` between steps for correct expansion accounting (works up to `v→c`).
- **CBH Gravity and capture**: the apparent horizon is found from the full LTB condition `2G·M(<r)/(c²r) + Λr²/3 = 1`. Matter and in-flight photon energy inside the inner apparent horizon are captured by the CBH.
- **Relativistic laser model ("photon rocket")**: SR and GR effects are strictly accounted for:
  - Rest mass loss includes relativistic dilation (`1/γ`) and gravitational time dilation at the source (`√f`).
  - Coordinate photon energy (contribution to BH mass) accounts for the Doppler effect (`γ(1−β_r)`) and gravitational shift.
  - During flight, photons experience cosmological redshift `a(t_emit)/a(t_arrival)`.

### Dust Matter (Dark + Baryonic)
- Dark and baryonic matter are modeled as a single dust component with density `ρ_m(t)`. Baryonic physics (recombination, pressure, etc.) is not modeled.

### Matter Visualization
- 2D display of matter points projected onto the screen. Colors distinguish the position of points relative to horizons.

### Cosmological Horizons and the Nariai Limit
- 4 horizons are tracked: LTB Hubble, LTB Event, empty-universe de Sitter reference, and Particle.
- **LTB Hubble / outer apparent horizon** is the outermost root of `g(r)=0`, where `g(r) = 2G·M(<r)/(c²r) + Λr²/3 − 1`.
- **LTB Event horizon** is computed separately as a radial null separatrix by integrating `dr/dt = R_dot − c` backward from a late-time LTB outer apparent horizon.
- **BH Event** in the visualization is the inner LTB apparent horizon, not a vacuum SdS cap.
- The project still reports the classic vacuum Nariai mass `M_N = c² / (3G√Λ)` as a reference scale. It is not used as a dynamic hard cap for the LTB horizon evolution.

## Constraints

The simulation uses simplified models and 2D visualization. Conscious deviations from strict cosmology include:

- **Radiation epoch is ignored** (`Ω_r ≈ 0`). The simulation starts deep in the matter-dominated epoch ($t=1$ billion years, $z \approx 5.7$), where the relative contribution of radiation is negligible.
- **No N-body structure formation**: matter points do not gravitationally interact with each other as galaxies or clusters. Their mass-energy is included in the spherical LTB `M(<r)` profile used for horizons.
- **LTB event horizon is predictive**: a true event horizon is global and depends on the future spacetime. The live simulation estimates it by evolving the current LTB mass profile forward and integrating the null separatrix backward.
- **2D visualization for 3D calculation**: vectors are calculated in 3D (radially) but rendered as a projection on the screen.
