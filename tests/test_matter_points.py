import numpy as np

import config
from physics import matter_points as matter_points_mod
from physics.cosmology import LambdaCDM
from physics.matter_points import MatterPoints
from utils.config_utils import get_collapse_start_time_seconds
from utils.constants import G, c
from utils.cosmology_utils import calculate_scale_factor_at_time


def test_hubble_drag_rescales_peculiar_momentum_inversely_with_scale_factor():
    """
    Проверка Hubble drag в обновлении скоростей.

    Для свободной массивной частицы в FRW сохраняется a·γ·v_pec = const,
    то есть peculiar momentum p_pec = γ·v_pec ∝ 1/a. В нерелятивистском
    пределе это даёт v_pec_phys ∝ 1/a (а не v_pec_phys = const).
    """
    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    config.MATTER_THRUST_POWER_PER_KG_W = 0.0
    try:
        matter_points = MatterPoints()
        old_scale_factor = 0.5
        new_scale_factor = 1.0
        physical_velocity = np.array([[1000.0, 0.0, 0.0]], dtype=np.float64)

        matter_points.points_comoving = np.array([[1.0e20, 0.0, 0.0]], dtype=np.float64)
        matter_points.velocities_comoving = physical_velocity / old_scale_factor
        matter_points.masses_per_point = np.array([1.0e30], dtype=np.float64)
        matter_points._previous_scale_factor = old_scale_factor
        matter_points._update_comoving_distances()

        matter_points.update_positions_and_velocities(
            dt=1.0,
            scale_factor=new_scale_factor,
            scale_ratio=1.0,
            r_black_hole=None,
            universe_time_seconds=1.0,
        )

        # Hubble drag (нерелятивистский предел): v_phys ∝ 1/a, то есть
        # v_phys_new = (a_old / a_new) · v_phys_old.
        expected_v_phys = (old_scale_factor / new_scale_factor) * physical_velocity
        np.testing.assert_allclose(
            matter_points.velocities_comoving * new_scale_factor,
            expected_v_phys,
            rtol=1e-10,
        )
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec


def _setup_one_emitter(thrust_w_per_kg, beta_radial, m_rest):
    """
    Создать MatterPoints с одной точкой-эмиттером, движущейся радиально с
    заданным β_radial (положительный = наружу), m_rest на одну точку.
    Лазерный луч направлен к центру.
    """
    matter_points = MatterPoints()
    # Точка на расстоянии 1 ly в +x.
    r = 9.461e15
    matter_points.points_comoving = np.array([[r, 0.0, 0.0]], dtype=np.float64)
    # v_phys = β·c в +x → v_comoving = v_phys / a = β·c (a=1).
    v_phys = np.array([[beta_radial * c, 0.0, 0.0]], dtype=np.float64)
    matter_points.velocities_comoving = v_phys.copy()
    matter_points.masses_per_point = np.array([m_rest], dtype=np.float64)
    matter_points._previous_scale_factor = 1.0
    matter_points._update_comoving_distances()
    # Явно делаем единственную точку лазерным эмиттером (минуя долевую логику).
    matter_points.laser_emitter_mask = np.array([True], dtype=bool)
    return matter_points


def _expected_laser_dm(thrust_w_per_kg, m_rest, dt, gamma=1.0, sqrt_lapse=1.0):
    """Непрерывное выгорание доступной массы при постоянных gamma и sqrt_lapse."""
    exponent = thrust_w_per_kg * dt * sqrt_lapse / (gamma * c * c)
    return m_rest * (-np.expm1(-exponent))


def test_laser_relativistic_corrections_at_high_beta():
    """
    Релятивистская модель «фотонной ракеты» в лаб. системе:
      • Потеря массы покоя за лаб. dt интегрируется экспоненциально
        с k = s/(γc²).
      • Энергия фотона в лаб. системе: E_phot_lab/c² = γ(1−β_r)·dm_rest.
      • Лаб. dp/dt = (1−β_r)·s·m/c (следствие первых двух).
    Тест проверяет одну точку, движущуюся наружу с β=0.6 (γ=1.25). Используем
    высокую удельную мощность s=1e10 Вт/кг, чтобы dm ≫ eps·m и FP-точность
    не маскировала проверку.
    """
    s = 1.0e10
    beta = 0.6
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    m_rest = 1.0
    dt = 1.0

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    original_eff = getattr(config, "MATTER_LASER_REMAINING_FRACTION", 1e-4)
    config.MATTER_THRUST_POWER_PER_KG_W = s
    config.MATTER_LASER_REMAINING_FRACTION = 0.0
    try:
        matter_points = _setup_one_emitter(s, beta, m_rest)

        # Импульс начала: p = γ·v_phys, без массы (как в коде).
        p_old_per_m = gamma * beta * c

        matter_points.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=None,
            universe_time_seconds=1.0,
        )

        # 1) Потеря массы покоя за dt: интеграл dm/dt = -k*m.
        m_new = matter_points.masses_per_point[0]
        dm_expected = _expected_laser_dm(s, m_rest, dt, gamma=gamma)
        np.testing.assert_allclose(m_rest - m_new, dm_expected, rtol=1e-9)

        # 2) Энергия фотона в лаб. на момент эмиссии: γ(1−β)·dm_rest.
        photon_mass = matter_points._laser_photon_mass_emit_kg
        assert photon_mass.shape == (1,)
        expected_photon_mass = gamma * (1.0 - beta) * dm_expected
        np.testing.assert_allclose(photon_mass[0], expected_photon_mass, rtol=1e-9)

        # 3) Прирост γv = (s/c)·dt (proper acceleration). Это инвариант
        # формулировки d(γv)/dt = s/c — лаб. dp/dt получит (1−β)·s·m/c
        # автоматически из совокупности обновлений γv и массы.
        v_new = matter_points.velocities_comoving[0, 0] * 1.0  # scale_factor=1
        beta2_new = (v_new / c) ** 2
        gamma_new = 1.0 / np.sqrt(1.0 - beta2_new)
        delta_gamma_v = gamma_new * v_new - p_old_per_m
        dm_linear = s * m_rest * dt / (gamma * c * c)
        active_fraction = dm_expected / dm_linear
        np.testing.assert_allclose(delta_gamma_v, s * dt * active_fraction / c, rtol=1e-9)
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec
        config.MATTER_LASER_REMAINING_FRACTION = original_eff


def test_laser_blueshifts_for_infalling_source():
    """
    Точка, падающая К центру (β_radial < 0), излучает фотон К центру —
    т.е. ВПЕРЁД относительно своего движения. В лаб. системе фотон
    blueshift-нут: E_phot_lab/c² = γ(1−β_r)·dm_rest > dm_rest при β_r < 0.
    """
    s = 1.0e10
    beta = -0.5
    gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    m_rest = 1.0
    dt = 1.0

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    original_eff = getattr(config, "MATTER_LASER_REMAINING_FRACTION", 1e-4)
    config.MATTER_THRUST_POWER_PER_KG_W = s
    config.MATTER_LASER_REMAINING_FRACTION = 0.0
    try:
        matter_points = _setup_one_emitter(s, beta, m_rest)

        matter_points.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=None,
            universe_time_seconds=1.0,
        )

        m_new = matter_points.masses_per_point[0]
        dm = m_rest - m_new

        photon_mass = matter_points._laser_photon_mass_emit_kg[0]
        # Blueshift: фактор γ(1−β) > 1 при β < 0.
        factor = gamma * (1.0 - beta)
        assert factor > 1.0
        np.testing.assert_allclose(photon_mass, factor * dm, rtol=1e-9)
        assert photon_mass > dm
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec
        config.MATTER_LASER_REMAINING_FRACTION = original_eff


def test_laser_relativistic_corrections_reduce_to_classical_at_low_beta():
    """
    В нерелятивистском пределе (β ≪ 1) γ → 1, (1−β) → 1, и новые формулы
    сводятся к классическим: dm = s·m·dt/c², E_phot/c² = dm.
    """
    s = 1.0e10
    beta = 1e-5
    m_rest = 1.0
    dt = 1.0

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    original_eff = getattr(config, "MATTER_LASER_REMAINING_FRACTION", 1e-4)
    config.MATTER_THRUST_POWER_PER_KG_W = s
    config.MATTER_LASER_REMAINING_FRACTION = 0.0
    try:
        matter_points = _setup_one_emitter(s, beta, m_rest)

        matter_points.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=None,
            universe_time_seconds=1.0,
        )

        dm = m_rest - matter_points.masses_per_point[0]
        photon_mass = matter_points._laser_photon_mass_emit_kg[0]

        # Классические значения и совпадение между mass_loss и photon mass
        # с относительной точностью O(β) ~ 1e-5.
        np.testing.assert_allclose(dm, _expected_laser_dm(s, m_rest, dt), rtol=1e-4)
        np.testing.assert_allclose(photon_mass, dm, rtol=1e-4)
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec
        config.MATTER_LASER_REMAINING_FRACTION = original_eff


def test_laser_high_power_burns_exponentially_within_one_dt():
    """
    Если s*dt/c² = 1, за шаг должна выгореть не вся масса и не линейные 100%,
    а интеграл непрерывного выгорания: 1 - exp(-1).
    """
    s = c * c
    m_rest = 1.0
    dt = 1.0

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    original_eff = getattr(config, "MATTER_LASER_REMAINING_FRACTION", 1e-4)
    config.MATTER_THRUST_POWER_PER_KG_W = s
    config.MATTER_LASER_REMAINING_FRACTION = 0.0
    try:
        matter_points = _setup_one_emitter(s, beta_radial=0.0, m_rest=m_rest)

        matter_points.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=None,
            universe_time_seconds=1.0,
        )

        expected_dm = 1.0 - np.exp(-1.0)
        dm = m_rest - matter_points.masses_per_point[0]
        np.testing.assert_allclose(dm, expected_dm, rtol=1e-12)
        np.testing.assert_allclose(matter_points._laser_photon_mass_emit_kg[0], expected_dm, rtol=1e-12)
        np.testing.assert_allclose(matter_points.masses_per_point[0], np.exp(-1.0), rtol=1e-12)
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec
        config.MATTER_LASER_REMAINING_FRACTION = original_eff


def _setup_one_emitter_with_bh(thrust_w_per_kg, r_phys, m_bh_kg, m_rest=1.0):
    """
    Создать MatterPoints с одной точкой-эмиттером в покое (β=0) на расстоянии
    r_phys от центра, при заданной массе центральной ЧД. Лазер работает.
    """
    matter_points = MatterPoints()
    matter_points.points_comoving = np.array([[r_phys, 0.0, 0.0]], dtype=np.float64)
    matter_points.velocities_comoving = np.zeros((1, 3), dtype=np.float64)
    matter_points.masses_per_point = np.array([m_rest], dtype=np.float64)
    matter_points._previous_scale_factor = 1.0
    matter_points._update_comoving_distances()
    matter_points.laser_emitter_mask = np.array([True], dtype=bool)
    matter_points.accumulated_bh_mass = float(m_bh_kg)
    return matter_points


def test_sds_gravitational_time_dilation_slows_laser_burn_near_bh():
    """
    Полная SdS-геодезика: dm_rest/dt_lab = −s·m·η/(γc²)·√f(r), где
    f(r) = 1 − r_s/r. Для покоящейся точки (β=0, γ=1) при r = 4·r_s
    ожидаем √f = √(3/4), и темп выгорания массы в (√(3/4) ≈ 0.866) раз
    меньше, чем без релятивистского лапса.
    """
    s = 1.0e10
    m_rest = 1.0
    dt = 1.0
    m_bh = 1.0e30
    r_s = 2.0 * G * m_bh / (c * c)
    r_phys = 4.0 * r_s
    expected_sqrt_f = np.sqrt(1.0 - r_s / r_phys)

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    original_eff = getattr(config, "MATTER_LASER_REMAINING_FRACTION", 1e-4)
    config.MATTER_THRUST_POWER_PER_KG_W = s
    config.MATTER_LASER_REMAINING_FRACTION = 0.0
    try:
        mp = _setup_one_emitter_with_bh(s, r_phys, m_bh, m_rest)
        mp.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=r_s,
            universe_time_seconds=1.0,
        )

        dm = m_rest - mp.masses_per_point[0]
        # γ=1 при β=0 → dm = m·(1-exp(-s·dt·√f/c²)).
        expected_dm = _expected_laser_dm(s, m_rest, dt, sqrt_lapse=expected_sqrt_f)
        np.testing.assert_allclose(dm, expected_dm, rtol=1e-9)

        # Координатная масса фотона = γ(1−β)·dm·√f = 1·1·dm·√f = dm·√f.
        photon_mass_coord = mp._laser_photon_mass_emit_kg[0]
        np.testing.assert_allclose(photon_mass_coord, dm * expected_sqrt_f, rtol=1e-9)
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec
        config.MATTER_LASER_REMAINING_FRACTION = original_eff


def test_sds_gravity_weakens_near_horizon():
    """
    SdS-радиальная геодезика для пробной частицы в координатном времени:
    d(γv)/dt = −(GM/r²)·√f/γ. Для покоящейся точки (γ=1) на r = 4·r_s
    ожидаем приращение γv = −(GM/r²)·√f·dt с фактором √f относительно
    нерелятивистского −GM/r²·dt.
    """
    m_rest = 1.0
    dt = 1.0
    m_bh = 1.0e30
    r_s = 2.0 * G * m_bh / (c * c)
    r_phys = 4.0 * r_s
    expected_sqrt_f = np.sqrt(1.0 - r_s / r_phys)

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    config.MATTER_THRUST_POWER_PER_KG_W = 0.0  # лазер выключен
    try:
        mp = _setup_one_emitter_with_bh(0.0, r_phys, m_bh, m_rest)
        mp.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=r_s,
            universe_time_seconds=1.0,
        )

        # v_phys_new = v_comoving_new (a=1). До шага v=0; после — отрицательная
        # радиальная (точка падает). |Δ(γv)| = (GM/r²)·√f·dt.
        v_after = mp.velocities_comoving[0]
        gamma_after = 1.0 / np.sqrt(1.0 - np.einsum('i,i', v_after, v_after) / (c * c))
        gv_radial = gamma_after * v_after[0]  # радиально на ось +x
        expected_dgv = -(G * m_bh / (r_phys * r_phys)) * expected_sqrt_f * dt
        np.testing.assert_allclose(gv_radial, expected_dgv, rtol=1e-6)
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec


def test_sds_reduces_to_newton_far_from_bh():
    """
    На r ≫ r_s имеем √f → 1, и SdS-формулы сводятся к ньютоновским:
    a = −GM/r², dm = m·(1-exp(-s·dt/(γc²))) (без gravitational time dilation).
    """
    s = 1.0e10
    m_rest = 1.0
    dt = 1.0
    m_bh = 1.0e30
    r_s = 2.0 * G * m_bh / (c * c)
    r_phys = 1.0e8 * r_s  # r ≫ r_s → √f − 1 ≈ −5e-9

    original_spec = config.MATTER_THRUST_POWER_PER_KG_W
    original_eff = getattr(config, "MATTER_LASER_REMAINING_FRACTION", 1e-4)
    config.MATTER_THRUST_POWER_PER_KG_W = s
    config.MATTER_LASER_REMAINING_FRACTION = 0.0
    try:
        mp = _setup_one_emitter_with_bh(s, r_phys, m_bh, m_rest)
        mp.update_positions_and_velocities(
            dt=dt,
            scale_factor=1.0,
            scale_ratio=1.0,
            r_black_hole=r_s,
            universe_time_seconds=1.0,
        )

        dm = m_rest - mp.masses_per_point[0]
        np.testing.assert_allclose(dm, _expected_laser_dm(s, m_rest, dt), rtol=1e-7)
    finally:
        config.MATTER_THRUST_POWER_PER_KG_W = original_spec
        config.MATTER_LASER_REMAINING_FRACTION = original_eff


def test_emitter_mask_selects_points_inside_event_horizon():
    """
    Авто-маска эмиттеров: True для точек, чьё комовинг-расстояние меньше
    χ_event(t_collapse), False — для точек снаружи.

    Размещаем 4 точки на радиусах: 0.5·R, 0.9·R, 1.05·R, 2·R, где
    R = χ_event(t_collapse). Эмиттерами становятся только первые две.
    """
    mp = MatterPoints()
    mp.points_comoving = np.array(
        [[1.0, 0.0, 0.0]], dtype=np.float64
    )
    chi_event = mp._emitter_comoving_radius_m()
    assert chi_event > 0.0, "χ_event должен быть положительным"

    radii = np.array([0.5, 0.9, 1.05, 2.0]) * chi_event
    points = np.zeros((len(radii), 3), dtype=np.float64)
    points[:, 0] = radii
    mp.points_comoving = points
    mp._update_comoving_distances()

    mp.init_laser_emitter_mask(len(points))
    np.testing.assert_array_equal(
        mp.laser_emitter_mask,
        np.array([True, True, False, False]),
    )


def test_emitter_radius_matches_cosmological_event_horizon():
    """
    χ_event, рассчитанный лазер-маской из η(t)-сетки matter_points,
    должен совпадать с комовинг-радиусом cosmological event horizon
    из physics/cosmology.py в момент t_collapse (с относительной точностью
    лучше 1%, ограниченной шагом η-сетки и обрезанием на t_max=200 Гyr).
    """
    chi_from_mask = MatterPoints._emitter_comoving_radius_m()

    cosmo = LambdaCDM()
    t_collapse_sec = get_collapse_start_time_seconds()
    t_collapse_yr = float(getattr(config, "LASER_START_TIME_YEARS", 0.0))
    a_collapse = calculate_scale_factor_at_time(t_collapse_yr)

    prev_a = cosmo.scale_factor
    cosmo.scale_factor = a_collapse
    try:
        r_event_phys = cosmo.cosmological_event_horizon(t_collapse_sec, 0.0)
    finally:
        cosmo.scale_factor = prev_a
    chi_event_cosmology = r_event_phys / max(a_collapse, 1e-300)

    np.testing.assert_allclose(chi_from_mask, chi_event_cosmology, rtol=1e-2)
