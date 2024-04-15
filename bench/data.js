window.BENCHMARK_DATA = {
  "lastUpdate": 1713209234246,
  "repoUrl": "https://github.com/ChristopherMayes/Xopt",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "rroussel@slac.stanford.edu",
            "name": "Ryan Roussel",
            "username": "roussel-ryan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a613b00abdb1ff1809a10b82096947803834cec5",
          "message": "Merge pull request #205 from ChristopherMayes/select_best_improvements\n\nadd returning best param set to vocs method,",
          "timestamp": "2024-04-15T14:51:35-04:00",
          "tree_id": "e03b89bd5a6bb1e18518373a543238322896812f",
          "url": "https://github.com/ChristopherMayes/Xopt/commit/a613b00abdb1ff1809a10b82096947803834cec5"
        },
        "date": 1713209160959,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_mo.py::test_mobo[0]",
            "value": 0.35037279208079836,
            "unit": "iter/sec",
            "range": "stddev: 0.6065093321700616",
            "extra": "mean: 2.854102894409088 sec\nrounds: 22"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[1]",
            "value": 0.3366625616737078,
            "unit": "iter/sec",
            "range": "stddev: 0.5703787651746188",
            "extra": "mean: 2.9703332471199952 sec\nrounds: 25"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[2]",
            "value": 0.3220832514189142,
            "unit": "iter/sec",
            "range": "stddev: 0.39404981690288243",
            "extra": "mean: 3.1047873355555535 sec\nrounds: 18"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[3]",
            "value": 0.28176520768996666,
            "unit": "iter/sec",
            "range": "stddev: 0.5551694056882699",
            "extra": "mean: 3.549054222124987 sec\nrounds: 16"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[4]",
            "value": 0.24430034058235267,
            "unit": "iter/sec",
            "range": "stddev: 0.8828550339747524",
            "extra": "mean: 4.0933221690000225 sec\nrounds: 17"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[5]",
            "value": 0.16634121935870952,
            "unit": "iter/sec",
            "range": "stddev: 1.4625241738807315",
            "extra": "mean: 6.011739025692315 sec\nrounds: 13"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[6]",
            "value": 0.13722149534620628,
            "unit": "iter/sec",
            "range": "stddev: 1.9843541763751622",
            "extra": "mean: 7.287487995062477 sec\nrounds: 16"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[7]",
            "value": 0.12958021671200518,
            "unit": "iter/sec",
            "range": "stddev: 2.0091781774130726",
            "extra": "mean: 7.717227408428569 sec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[8]",
            "value": 0.13195757061267266,
            "unit": "iter/sec",
            "range": "stddev: 1.724502029576103",
            "extra": "mean: 7.578193470500011 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[9]",
            "value": 0.11785834555252148,
            "unit": "iter/sec",
            "range": "stddev: 2.319177741474722",
            "extra": "mean: 8.484761900500018 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[10]",
            "value": 0.10337687019788788,
            "unit": "iter/sec",
            "range": "stddev: 2.3035257768657385",
            "extra": "mean: 9.673343738166599 sec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[11]",
            "value": 0.07137873985349816,
            "unit": "iter/sec",
            "range": "stddev: 1.6442433056712462",
            "extra": "mean: 14.009773807333355 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[12]",
            "value": 0.3737443762093363,
            "unit": "iter/sec",
            "range": "stddev: 0.3415998421257167",
            "extra": "mean: 2.6756255442353316 sec\nrounds: 17"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[13]",
            "value": 0.4035921308185588,
            "unit": "iter/sec",
            "range": "stddev: 0.31957785205678607",
            "extra": "mean: 2.477749003608709 sec\nrounds: 23"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[14]",
            "value": 0.36041833284285685,
            "unit": "iter/sec",
            "range": "stddev: 0.7269910608821396",
            "extra": "mean: 2.774553647458333 sec\nrounds: 24"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[15]",
            "value": 0.2719974782175414,
            "unit": "iter/sec",
            "range": "stddev: 1.2255414516236316",
            "extra": "mean: 3.6765046740624854 sec\nrounds: 16"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[16]",
            "value": 0.2871763894298659,
            "unit": "iter/sec",
            "range": "stddev: 0.7611499978746262",
            "extra": "mean: 3.482180418749988 sec\nrounds: 20"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[17]",
            "value": 0.21872695821466207,
            "unit": "iter/sec",
            "range": "stddev: 0.8012974365913429",
            "extra": "mean: 4.571910148444456 sec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[18]",
            "value": 0.2226858372497415,
            "unit": "iter/sec",
            "range": "stddev: 2.3263136834967004",
            "extra": "mean: 4.490631341222222 sec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[19]",
            "value": 0.16761050817908255,
            "unit": "iter/sec",
            "range": "stddev: 3.9725465258294137",
            "extra": "mean: 5.966213042750013 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[20]",
            "value": 0.17821986954557334,
            "unit": "iter/sec",
            "range": "stddev: 2.2118886971302842",
            "extra": "mean: 5.611046638906252 sec\nrounds: 32"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[21]",
            "value": 0.06913125984847505,
            "unit": "iter/sec",
            "range": "stddev: 8.060939547676481",
            "extra": "mean: 14.465236163666683 sec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[22]",
            "value": 0.08134362276579395,
            "unit": "iter/sec",
            "range": "stddev: 5.4852453528849345",
            "extra": "mean: 12.293526720333299 sec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[23]",
            "value": 0.11338155776352317,
            "unit": "iter/sec",
            "range": "stddev: 1.6937746235874334",
            "extra": "mean: 8.819776511499981 sec\nrounds: 4"
          }
        ]
      }
    ]
  }
}