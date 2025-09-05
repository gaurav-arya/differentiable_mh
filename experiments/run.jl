#!/bin/bash
#
#SBATCH -J naiss2024-22-1100-dmh
#SBATCH -t 64:00:00
#SBATCH --mem=16000
#SBATCH -n 1
#SBATCH --array=0-3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rubense@chalmers.se

#=
tasks=(
    "data_contamination/analyze_data_contamination.jl"
    "prior_sensitivity/analyze_prior_sensitivity_problem.jl"
    "rwmh_tuning/analyze_mh_tuning_problem.jl"
    "conditional_sde/analyze_conditional_sde.jl"
)

module add julia/1.10.2-bdist
export JULIA_DEPOT_PATH="/proj/pdmps/julia:$JULIA_DEPOT_PATH"
export JULIA_PROJECT=/proj/pdmps/repos/dmh/experiments
export OPENBLAS_NUM_THREADS=1
cd /proj/pdmps/repos/dmh/experiments
echo "Running task ${tasks[$SLURM_ARRAY_TASK_ID]}"
exec julia /proj/pdmps/repos/dmh/experiments/run.jl ${tasks[$SLURM_ARRAY_TASK_ID]}
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