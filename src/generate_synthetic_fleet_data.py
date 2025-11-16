import numpy as np
import pandas as pd

def generate_fleet_data(n_vehicles: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    vehicle_ids = np.arange(1, n_vehicles + 1)

    # Basic vehicle attributes / Podstawowe cechy pojazdów
    age_years = rng.integers(0, 15, size=n_vehicles)  # vehicle age / wiek pojazdu w latach
    annual_mileage = rng.integers(5_000, 60_000, size=n_vehicles)   # roczny przebieg

    #  Rodzaj użytkowania pojazdu
    usage_type = rng.choice(
        ["delivery", "taxi", "sales", "service"],
        size=n_vehicles,
        p=[0.35, 0.25, 0.25, 0.15]
    )

    # Region użytkowania
    region = rng.choice(
        ["urban", "suburban", "rural"],
        size=n_vehicles,
        p=[0.5, 0.3, 0.2]
    )

    # Base incident rate (lambda) depending on features  / Bazowy poziom ryzyka incydentów (częstość)
    base_lambda = 0.15  # baseline incidents per year / średnia liczba incydentów na pojazd

    lambda_age = 0.02 * age_years # Wpływ wieku na ryzyko incydentów
    lambda_mileage = (annual_mileage - 20000) / 50000 * 0.1  # more mileage -> more incidents

    lambda_usage = np.select(
        [
            usage_type == "delivery",
            usage_type == "taxi",
            usage_type == "sales",
            usage_type == "service",
        ],
        [0.10, 0.12, 0.05, 0.03],
        default=0.0,
    )

    lambda_region = np.select(
        [
            region == "urban",
            region == "suburban",
            region == "rural",
        ],
        [0.08, 0.04, 0.02],
        default=0.0,
    )

    incident_rate = np.clip(
        base_lambda + lambda_age + lambda_mileage + lambda_usage + lambda_region,
        0.01,
        None,
    )

    # Number of incidents ~ Poisson / Liczba incydentów generowana z rozkładu Poissona
    n_incidents = rng.poisson(lam=incident_rate)

    # Cost per incident (severity), only for vehicles with incidents  / Współczynnik kosztu zależny od wieku i przebiegu
    base_cost = 500.0
    severity_multiplier = 1.0 + 0.03 * age_years + 0.00001 * annual_mileage

    avg_incident_cost = base_cost * severity_multiplier

    # total repair cost per year / Całkowity koszt napraw w roku (lognormal → duża zmienność)
    total_repair_cost = n_incidents * rng.lognormal(
        mean=np.log(avg_incident_cost),
        sigma=0.4,
        size=n_vehicles
    )

    # small noise, some zero-cost even with incidents (warranty etc.)
    mask_zero_cost = (n_incidents > 0) & (rng.random(n_vehicles) < 0.1)
    total_repair_cost[mask_zero_cost] = 0.0

    # Create table / Tworzenie ramki danych
    df = pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "age_years": age_years,
            "annual_mileage": annual_mileage,
            "usage_type": usage_type,
            "region": region,
            "incident_rate_theoretical": incident_rate,
            "n_incidents": n_incidents,
            "total_repair_cost": total_repair_cost,
        }
    )

    return df

# Save csv / Zapis do pliku csv
if __name__ == "__main__":
    df = generate_fleet_data()
    df.to_csv("data/fleet_incidents_synthetic.csv", index=False)
    print("Saved data/fleet_incidents_synthetic.csv")
