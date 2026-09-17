# AI-Enabled Intelligent Traffic Management System

An AI-assisted prototype for adaptive traffic-signal control at a four-way urban intersection. The system combines traffic simulation, vehicle-count aggregation, congestion forecasting, and a DQN-based signal controller. It compares the AI policy with a fixed-timer baseline and generates a browser-based performance dashboard.

## Features

- Synthetic four-way traffic simulation with normal, peak, emergency, and accident scenarios
- Lane-level queue, waiting-time, throughput, vehicle-count, and signal-phase metrics
- Sequence-based congestion and queue forecasting
- DQN signal controller with queue pressure, waiting time, emergency priority, and accident awareness
- SUMO GUI integration for observing traffic and signal phases in real time
- Static HTML and optional Streamlit dashboards
- Privacy-aware design that stores aggregate counts rather than identifiable video

## Architecture

```text
Traffic simulation / sensors
            |
            v
Aggregate vehicle counts and lane metrics
            |
            +--> Queue and congestion forecaster
            |
            +--> Emergency and accident checks
                          |
                          v
                 DQN signal controller
                          |
                          v
                  Adaptive signal phase
                          |
                          v
                    Reports / dashboard
```

## Project Structure

```text
.
|-- #Main/
|   |-- quick_sumo_test.py
|   |-- run_sumo_gui.py
|   `-- train_on_csv_and_run.py
|-- src/
|   |-- control/       DQN controller and policy logic
|   |-- dashboard/     HTML and Streamlit dashboard helpers
|   |-- detection/     Aggregate vehicle-count processing
|   |-- prediction/    Queue and congestion forecasting
|   `-- simulation/    Synthetic traffic and SUMO environments
|-- data/processed/    Generated and prepared CSV metrics
|-- reports/           Dashboard and performance outputs
|-- ITMS_Report.pdf    Technical report
`-- README.md
```

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- SUMO and TraCI for SUMO GUI execution
- Streamlit is optional and only required for the interactive dashboard

Install the Python packages from the repository root:

```bash
python -m pip install numpy pandas traci streamlit
```

For SUMO, install the desktop package for your operating system and set `SUMO_HOME` to the SUMO installation directory. Ensure `sumo` and `sumo-gui` are available on your `PATH`.

## Run The Project

Run the short SUMO smoke test:

```bash
cd "#Main"
python quick_sumo_test.py
```

Train the controller on synthetic scenarios and launch the SUMO emergency scenario:

```bash
cd "#Main"
python run_sumo_gui.py
```

Train from the processed fixed-timer CSV data, launch SUMO, and generate a dashboard after the run:

```bash
cd "#Main"
python train_on_csv_and_run.py
```

The SUMO window can be paused with `Space`, sped up with `F3`, slowed down with `F4`, or closed to stop the simulation. The generated static dashboard is written to `reports/dashboard.html`.

Run the optional Streamlit dashboard from the repository root:

```bash
streamlit run src/dashboard/app.py
```

## Results

The evaluation reported in `ITMS_Report.pdf` compares the fixed-timer baseline with the AI controller on identical demand streams:

| Scenario | Average wait reduction | AI max queue | AI throughput |
| --- | ---: | ---: | ---: |
| Normal | 18.86% | 70 | 599 |
| Peak | 0.00% | 343 | 600 |
| Emergency | 6.53% | 126 | 598 |
| Accident | 2.91% | 179 | 544 |

Emergency clearance improved from 51 seconds with the fixed timer to 6 seconds with the AI controller. The measured decision loop remained below the 3-second requirement in all evaluated scenarios.

Peak-hour traffic reached simulated saturation, so the AI controller did not reduce average waiting time in that scenario. The prototype should therefore be treated as a simulation and research demonstrator, not as a production traffic-control system.

## Limitations And Future Work

- Connect the controller to a calibrated SUMO/TraCI road network and real origin-destination demand
- Replace simulation-only vehicle counting with a validated YOLO or OpenCV camera pipeline
- Evaluate weather, lighting, and camera-placement variations
- Validate deployment performance on edge hardware such as Raspberry Pi or NVIDIA Jetson
- Add stronger safety constraints and human-supervised controls before any real-world use

## Report

See [ITMS_Report.pdf](ITMS_Report.pdf) for the full technical report, diagrams, methodology, evaluation table, and references.

## Author

Muhammed Salah Hussain  
GitHub: [M-S-H-Git](https://github.com/M-S-H-Git)  
LinkedIn: [Muhammed Salah Hussain](https://linkedin.com/in/muhammed-salah-hussain-231797388)