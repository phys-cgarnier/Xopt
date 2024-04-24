window.BENCHMARK_DATA = {
  "lastUpdate": 1713972138372,
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
          "id": "76d4af6234ee7309bdc7bce6e44347dfc1550a55",
          "message": "Merge pull request #221 from t-bz/modularize_visualization\n\nModularize visualization",
          "timestamp": "2024-04-24T10:42:36-04:00",
          "tree_id": "a0b7339e3b211704d3e3d6783787a6dd366d3a4e",
          "url": "https://github.com/ChristopherMayes/Xopt/commit/76d4af6234ee7309bdc7bce6e44347dfc1550a55"
        },
        "date": 1713971769432,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_mo.py::test_mobo[0]",
            "value": 0.3676180782691217,
            "unit": "iter/sec",
            "range": "stddev: 0.39550202839700516",
            "extra": "mean: 2.7202144266363617 sec\nrounds: 22"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[1]",
            "value": 0.3415716962423977,
            "unit": "iter/sec",
            "range": "stddev: 0.5792096623670199",
            "extra": "mean: 2.927643042444436 sec\nrounds: 18"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[2]",
            "value": 0.2998279803738579,
            "unit": "iter/sec",
            "range": "stddev: 0.7078669551326882",
            "extra": "mean: 3.3352457590952387 sec\nrounds: 21"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[3]",
            "value": 0.3169403194648233,
            "unit": "iter/sec",
            "range": "stddev: 0.4974359298686503",
            "extra": "mean: 3.155168145499987 sec\nrounds: 18"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[4]",
            "value": 0.2607553209591136,
            "unit": "iter/sec",
            "range": "stddev: 0.6974261104011755",
            "extra": "mean: 3.835012824749988 sec\nrounds: 16"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[5]",
            "value": 0.172111232099904,
            "unit": "iter/sec",
            "range": "stddev: 1.0554929375614013",
            "extra": "mean: 5.810196044727274 sec\nrounds: 11"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[6]",
            "value": 0.14348485943402195,
            "unit": "iter/sec",
            "range": "stddev: 1.5539004236312892",
            "extra": "mean: 6.96937644811107 sec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[7]",
            "value": 0.13911891494528777,
            "unit": "iter/sec",
            "range": "stddev: 2.158056059567391",
            "extra": "mean: 7.188095165875012 sec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[8]",
            "value": 0.14617168485294088,
            "unit": "iter/sec",
            "range": "stddev: 2.720307161148527",
            "extra": "mean: 6.841270256999986 sec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[9]",
            "value": 0.13467908190508818,
            "unit": "iter/sec",
            "range": "stddev: 1.2679168143273705",
            "extra": "mean: 7.425058040599993 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[10]",
            "value": 0.10191682707236176,
            "unit": "iter/sec",
            "range": "stddev: 1.9216010521725413",
            "extra": "mean: 9.811922414833342 sec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[11]",
            "value": 0.06513538621418084,
            "unit": "iter/sec",
            "range": "stddev: 4.539727699222095",
            "extra": "mean: 15.352637914999983 sec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[12]",
            "value": 0.3754387144455646,
            "unit": "iter/sec",
            "range": "stddev: 0.5697831734144216",
            "extra": "mean: 2.6635505650416653 sec\nrounds: 24"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[13]",
            "value": 0.43428025541301335,
            "unit": "iter/sec",
            "range": "stddev: 0.3301046955149022",
            "extra": "mean: 2.302660522866669 sec\nrounds: 30"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[14]",
            "value": 0.38930794436695304,
            "unit": "iter/sec",
            "range": "stddev: 0.6393107063684996",
            "extra": "mean: 2.5686606566071566 sec\nrounds: 28"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[15]",
            "value": 0.30776998630524394,
            "unit": "iter/sec",
            "range": "stddev: 0.5393720921649474",
            "extra": "mean: 3.249179726733352 sec\nrounds: 15"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[16]",
            "value": 0.2955561832218716,
            "unit": "iter/sec",
            "range": "stddev: 0.965226219800976",
            "extra": "mean: 3.38345146123811 sec\nrounds: 21"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[17]",
            "value": 0.22492190987380942,
            "unit": "iter/sec",
            "range": "stddev: 1.1412782724342603",
            "extra": "mean: 4.445987500999977 sec\nrounds: 17"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[18]",
            "value": 0.18597823096557478,
            "unit": "iter/sec",
            "range": "stddev: 3.072952918654934",
            "extra": "mean: 5.376973395263145 sec\nrounds: 19"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[19]",
            "value": 0.16444079716771318,
            "unit": "iter/sec",
            "range": "stddev: 2.5429183659420667",
            "extra": "mean: 6.081215958714308 sec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[20]",
            "value": 0.15732530009912138,
            "unit": "iter/sec",
            "range": "stddev: 3.3759783275543827",
            "extra": "mean: 6.356256745545434 sec\nrounds: 11"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[21]",
            "value": 0.09132180574660444,
            "unit": "iter/sec",
            "range": "stddev: 2.3370875816932606",
            "extra": "mean: 10.95028719400002 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[22]",
            "value": 0.09325456137878979,
            "unit": "iter/sec",
            "range": "stddev: 4.319282084596866",
            "extra": "mean: 10.723336051500041 sec\nrounds: 4"
          },
          {
            "name": "benchmarks/test_mo.py::test_mobo[23]",
            "value": 0.07148669575456248,
            "unit": "iter/sec",
            "range": "stddev: 3.635189297996015",
            "extra": "mean: 13.988616895000035 sec\nrounds: 8"
          }
        ]
      }
    ]
  }
}