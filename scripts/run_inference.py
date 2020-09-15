from so_study.main import construct_parser, main
from so_study.badge_experiment_settings import *

if __name__ == '__main__':

    parser = construct_parser()
    args = parser.parse_args()

    if args.target_badge == "StrunkWhite":
        settings = StrunkWhiteExperiment
    elif args.target_badge == "CopyEditor":
        settings = CopyEditorExperiment
    elif args.target_badge == "CivicDuty":
        settings = CivicDutyExperiment
    elif args.target_badge == "Electorate":
        settings = ElectorateExperiment
    else:
        raise ValueError(f"No experiment settings for {args.target_badge} found")

    main(args, settings)
