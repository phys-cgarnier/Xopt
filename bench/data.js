window.BENCHMARK_DATA = {
  "lastUpdate": 1706722896118,
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
          "id": "f312347c792e86fb6df121cab52327298a5a82e4",
          "message": "Merge pull request #189 from ChristopherMayes/hessian_kernel\n\nimplement the kernel from Hessian in botorch",
          "timestamp": "2024-01-31T11:01:50-06:00",
          "tree_id": "7436dffa40404066b55b5de126fc0aa2bd14b067",
          "url": "https://github.com/ChristopherMayes/Xopt/commit/f312347c792e86fb6df121cab52327298a5a82e4"
        },
        "date": 1706722837329,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[0]",
            "value": 0.33372587950688753,
            "unit": "iter/sec",
            "range": "stddev: 0.6289702136753141",
            "extra": "mean: 2.9964712400416693 sec\nrounds: 24"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[1]",
            "value": 0.353249407586657,
            "unit": "iter/sec",
            "range": "stddev: 0.3677468187428776",
            "extra": "mean: 2.830861081500005 sec\nrounds: 22"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[2]",
            "value": 0.31252619795015546,
            "unit": "iter/sec",
            "range": "stddev: 0.5620241137745666",
            "extra": "mean: 3.1997317554782696 sec\nrounds: 23"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[3]",
            "value": 0.2916042287390292,
            "unit": "iter/sec",
            "range": "stddev: 0.6248202624810768",
            "extra": "mean: 3.42930554993751 sec\nrounds: 16"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[4]",
            "value": 0.24923013656281118,
            "unit": "iter/sec",
            "range": "stddev: 0.8020155175076278",
            "extra": "mean: 4.012355864307683 sec\nrounds: 13"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[5]",
            "value": 0.1712238335496661,
            "unit": "iter/sec",
            "range": "stddev: 0.7838337975420223",
            "extra": "mean: 5.840308438777798 sec\nrounds: 9"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[6]",
            "value": 0.14218278284657912,
            "unit": "iter/sec",
            "range": "stddev: 2.2477543319243294",
            "extra": "mean: 7.0332003634999865 sec\nrounds: 8"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[7]",
            "value": 0.12432114547957861,
            "unit": "iter/sec",
            "range": "stddev: 2.473053849381623",
            "extra": "mean: 8.043683929571444 sec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[8]",
            "value": 0.12832145047214186,
            "unit": "iter/sec",
            "range": "stddev: 1.8087350148997392",
            "extra": "mean: 7.792929368555544 sec\nrounds: 9"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[9]",
            "value": 0.11783242033855706,
            "unit": "iter/sec",
            "range": "stddev: 1.8053478165683587",
            "extra": "mean: 8.486628698000024 sec\nrounds: 12"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[10]",
            "value": 0.1232324072609631,
            "unit": "iter/sec",
            "range": "stddev: 2.9801682682953983",
            "extra": "mean: 8.114748565142852 sec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[11]",
            "value": 0.06591605398869672,
            "unit": "iter/sec",
            "range": "stddev: 3.3368616534269457",
            "extra": "mean: 15.170811046599965 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[12]",
            "value": 0.36720860687969575,
            "unit": "iter/sec",
            "range": "stddev: 0.4219571001640553",
            "extra": "mean: 2.723247716052631 sec\nrounds: 19"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[13]",
            "value": 0.38889359723987943,
            "unit": "iter/sec",
            "range": "stddev: 0.6660950114669858",
            "extra": "mean: 2.571397439035682 sec\nrounds: 28"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[14]",
            "value": 0.3851018636087709,
            "unit": "iter/sec",
            "range": "stddev: 0.33714314847187793",
            "extra": "mean: 2.5967155563181348 sec\nrounds: 22"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[15]",
            "value": 0.3283454573206669,
            "unit": "iter/sec",
            "range": "stddev: 0.4864566450620737",
            "extra": "mean: 3.0455728188235165 sec\nrounds: 17"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[16]",
            "value": 0.2790778084134023,
            "unit": "iter/sec",
            "range": "stddev: 1.1661907670059488",
            "extra": "mean: 3.5832300879999908 sec\nrounds: 21"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[17]",
            "value": 0.21228325351874872,
            "unit": "iter/sec",
            "range": "stddev: 1.0033500037125933",
            "extra": "mean: 4.710687175857141 sec\nrounds: 14"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[18]",
            "value": 0.23490114822757172,
            "unit": "iter/sec",
            "range": "stddev: 1.9315246232052037",
            "extra": "mean: 4.257109884500021 sec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[19]",
            "value": 0.13937657943088289,
            "unit": "iter/sec",
            "range": "stddev: 3.5463614961133607",
            "extra": "mean: 7.174806585749953 sec\nrounds: 12"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[20]",
            "value": 0.15037875304381007,
            "unit": "iter/sec",
            "range": "stddev: 2.0405294623192676",
            "extra": "mean: 6.64987559584743 sec\nrounds: 59"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[21]",
            "value": 0.0829443467057224,
            "unit": "iter/sec",
            "range": "stddev: 2.5132698315491155",
            "extra": "mean: 12.05627676528567 sec\nrounds: 7"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[22]",
            "value": 0.08440579954458088,
            "unit": "iter/sec",
            "range": "stddev: 4.860549916596097",
            "extra": "mean: 11.847527129599985 sec\nrounds: 5"
          },
          {
            "name": "tests/benchmarks/test_mo.py::test_mobo[23]",
            "value": 0.07226075815377554,
            "unit": "iter/sec",
            "range": "stddev: 2.666513110168338",
            "extra": "mean: 13.838769832333279 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}