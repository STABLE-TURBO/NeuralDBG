import sys
from click.testing import CliRunner

# Import the CLI entry
from neural.cli.cli import cli

def main():
    # First arg is output file to write results to
    out_file = sys.argv[1] if len(sys.argv) > 1 else 'cli_out.txt'
    # Remaining args are passed to the CLI
    args = sys.argv[2:]
    runner = CliRunner()
    result = runner.invoke(cli, args)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('EXIT=' + str(result.exit_code) + '\n')
        f.write(result.output)
        if result.exception:
            f.write('\nEXC=' + repr(result.exception))

if __name__ == '__main__':
    main()

