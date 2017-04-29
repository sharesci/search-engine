import { Component, Input, OnInit } from '@angular/core';
import { Router, NavigationStart } from '@angular/router';
import { AuthenticationService } from '../../services/authentication.service.js'
import { SearchService } from '../../services/search.service.js';
import { SharedService } from '../../services/shared.service.js';
import { ISearchResults } from '../../entities/search-results.interface.js';
import { Subject } from 'rxjs/Subject';
import 'rxjs/add/operator/filter';

@Component({
    selector: 'ss-navbar',
    templateUrl: 'src/app/components/navbar/navbar.component.html'
})

export class NavbarComponent {
    ishome: boolean = false;
    hideLoginBtn: boolean = false;
    searchToken: string = '';
    user: string = '';

    constructor(private _authenticationService : AuthenticationService,
                private _router: Router,
                private _searchService: SearchService, 
                private _sharedService: SharedService) { 
        _sharedService.isUserLoggedIn$
            .subscribe(                
                isUserLoggedIn => { 
                    this.hideLoginBtn = isUserLoggedIn;
                    this.user = localStorage.getItem("currentUser") || "";
                 }
        );
            
        _router.events
            .filter(event => event instanceof NavigationStart)
            .subscribe((event:NavigationStart) => {
                this.toggleSearchBox(event.url);
        });
        
        this.user = localStorage.getItem("currentUser") || "";
        this.hideLoginBtn = !!this.user;
    }

    toggleSearchBox(currenturl: string){
        if(currenturl == "/home" || currenturl == "/"){
            this.ishome = true;
        }
        else{
            this.ishome = false;
        }
    }

    toggleLoginBtn() {
        if(this.user) {
            this.hideLoginBtn = true;
        }
        else {
            this.user = "";
            this.hideLoginBtn = false;
        }
    }

    logout() {
        this._authenticationService.logout()
            .subscribe(null, null, () => {
                localStorage.removeItem('currentUser');
                this._sharedService.setLoginStatus(false);
        })
    }

    search() {
        this._searchService.search(this.searchToken)
            .map(response => <ISearchResults>response)
            .subscribe( 
                results => { this._sharedService.addSearchResults(results); 
                             this._sharedService.addSearchTerm(this.searchToken) },
                error => console.log(error)
            );
    }
}