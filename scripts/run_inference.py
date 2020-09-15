from so_study.main import construct_parser, main

if __name__ == '__main__':

    parser = construct_parser()
    args = parser.parse_args()
    main(args)
