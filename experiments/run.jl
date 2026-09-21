#!/bin/bash
#
#SBATCH -A C3SE2026-1-16 -p vera
#SBATCH -J dmh-experiments
#SBATCH -t 7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rubense@chalmers.se
#SBATCH --output=%x.%j.out

#=
tasks=(
    "data_contamination/analyze_data_contamination.jl"
    "prior_sensitivity/analyze_prior_sensitivity_problem.jl"
    "rwmh_tuning/analyze_mh_tuning_problem.jl"
)

module load Julia/1.10.2-linux-x86_64
export JULIA_NUM_THREADS=1
repo_dir="$REPOS/differentiable_mh"
experiment_dir="$repo_dir/experiments"
export JULIA_PROJECT="$experiment_dir"

cd "$experiment_dir"
echo "Running task ${tasks[$SLURM_ARRAY_TASK_ID]}"
exec julia "$experiment_dir/run.jl" "${tasks[$SLURM_ARRAY_TASK_ID]}"
=#
#
# Script ends here

using Literate

function preprocess(content)
    new_lines = map(split(content, "\n")) do line
        if endswith(line, "#src")
            line
        elseif startswith(line, "##cell")
            "#src"
        elseif startswith(line, "#text")
            replace(line, "#text" => "#")
        # try and save comments; strip necessary since Literate.jl also treats indented comments on their own line as markdown.
        elseif startswith(strip(line), "#") && !startswith(strip(line), "#=") && !startswith(strip(line), "#-")
            # TODO: should be replace first occurence only?
            replace(line, "#" => "##")
        # special to change loadpath
        elseif occursin("@__DIR__", line)
            replace(line, "@__DIR__" => "$(repr(dirname(abspath(first(ARGS)))))")
        else
            line
        end
    end
    return join(new_lines, "\n")
end

withenv("JULIA_DEBUG" => "Literate") do
    @time Literate.markdown(first(ARGS), joinpath(pwd(), "..", "docs", "src", "tutorials"); execute = true, flavor = Literate.CommonMarkFlavor(), preprocess = preprocess)
end
