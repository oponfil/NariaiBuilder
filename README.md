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
  - Hubble Horizon (r_H = c/H(t))
  - Cosmological Event Horizon (limitation due to accelerated expansion)
  - de Sitter Horizon (r_dS = c/√(Λ/3))
  - Particle Horizon (maximum distance light has traveled)
- Schwarzschild-de Sitter horizons for the central black hole
- Laser feeding of the central black hole with discrete photon packets
- Precomputed horizon caches to speed up the simulation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Before the first run (or after changing the time limits in `config.py`), you must precompute the cosmological horizons:

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
  - `matter_simulation.py` - generation and management of matter points
  - `matter_points.py` - dynamics of matter points, laser emitters, and photons
  - `mass_calculator.py` - calculation of masses inside horizons and the CBH
  - `nariai.py` - calculations of the Nariai limit (SdS) and "mass above background"
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
- `tests/` - tests for cosmological calculations and matter dynamics

## Physical Models

### ΛCDM and Expansion
- Flat ΛCDM model with matter and dark energy
- Planck 2018 + BAO parameters: H₀ = 67.4 km/s/Mpc, Ω_Λ = 0.685, Ω_DM = 0.266, Ω_B = 0.049
- Scale factor a(t) is calculated from the Friedmann equation
- Hubble parameter H(t)
- Precomputed tables with interpolation are used for horizons

### Matter Points and Laser
- **Strictly radial model**: matter is represented by discrete points moving strictly toward or away from the center (no angular momentum or orbits).
- **Laser emitter selection**: points inside the cosmological event horizon `χ < χ_event` are automatically selected. Photons from the rest will be carried away by expansion and will never reach the CBH.
- **Hubble drag**: the peculiar momentum of points scales as `p_pec ∝ 1/a` between steps for correct expansion accounting (works up to `v→c`).
- **CBH Gravity**: a hybrid model is used (ΛCDM background + full Schwarzschild-de Sitter radial geodesic). Gravitational time dilation (lapse `√f`) is accounted for strictly according to the SdS metric.
- **Relativistic laser model ("photon rocket")**: SR and GR effects are strictly accounted for:
  - Rest mass loss includes relativistic dilation (`1/γ`) and gravitational time dilation at the source (`√f`).
  - Coordinate photon energy (contribution to BH mass) accounts for the Doppler effect (`γ(1−β_r)`) and gravitational shift.
  - During flight, photons experience cosmological redshift `a(t_emit)/a(t_arrival)`.

### Dust Matter (Dark + Baryonic)
- Dark and baryonic matter are modeled as a single dust component with density `ρ_m(t)`. Baryonic physics (recombination, pressure, etc.) is not modeled.

### Matter Visualization
- 2D display of matter points projected onto the screen. Colors distinguish the position of points relative to horizons.

### Cosmological Horizons and the Nariai Limit
- 4 horizons are tracked: Hubble, Event, de Sitter, Particle.
- The project calculates the classic vacuum Nariai mass `M_N = c² / (3G√Λ)`. The CBH mass in the info panel is shown as a fraction of this limit.

## Constraints

The simulation uses simplified models and 2D visualization. Conscious deviations from strict cosmology include:

- **Radiation epoch is ignored** (`Ω_r ≈ 0`). The simulation starts deep in the matter-dominated epoch ($t=1$ billion years, $z \approx 5.7$), where the relative contribution of radiation is negligible.
- **No matter self-gravity**: points are attracted only to the CBH. Background density is accounted for in the Friedmann equation, but N-body interaction (galaxy formation) is not modeled.
- **Bypassing the "frozen-at-horizon" singularity**: in Schwarzschild coordinate time, a falling point infinitely approaches the horizon. To prevent the simulation from "freezing", gravitational time dilation (lapse) is bounded below (`√f_min = 0.01`). This allows the point to cross the horizon in a finite number of steps.
- **2D visualization for 3D calculation**: vectors are calculated in 3D (radially) but rendered as a projection on the screen.
