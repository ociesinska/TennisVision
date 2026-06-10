import argparse

from tennisvision.tasks.detection.experiments import DetectionExperimentConfig, run_experiment


def main() -> None:

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    cfg = DetectionExperimentConfig(
        backend=args.backend,
        model_name=args.model_name,
        data_config=args.data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
    )

    run_experiment(cfg)

if __name__ == "__main__":
    main()



    
