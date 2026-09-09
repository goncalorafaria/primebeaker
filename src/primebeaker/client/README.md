# PrimeBeaker client compatibility imports

Client implementations now live in the standalone `literegistry-tool-client`
companion package in the LiteRegistry repository. Import new code from
`literegistry_tool_client`; `primebeaker.client`, its service modules, and
`primebeaker.clients` continue to re-export the same classes.

See [the client package documentation](https://github.com/goncalorafaria/literegistry/tree/main/literegistry_tool_client).
PrimeBeaker installs the published companion as a dependency. No Beaker submission behavior changes.
