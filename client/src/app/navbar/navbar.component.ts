import { Component, Input } from '@angular/core';
import { Router, NavigationStart } from '@angular/router';
import { AuthenticationService } from '../services/authentication.service.js'
import 'rxjs/add/operator/filter';

@Component({
    selector: 'ss-navbar',
    templateUrl: 'src/app/navbar/navbar.component.html'
})

export class NavbarComponent {
    ishome: boolean = false;
    hideLoginBtn: boolean = false;

    constructor(private _authenticationService: AuthenticationService, private _router: Router) { 
        _authenticationService.isUserLoggedIn$
            .subscribe(
                isUserLoggedIn => { this.hideLoginBtn = isUserLoggedIn }
        );
            
        _router.events
            .filter(event => event instanceof NavigationStart)
            .subscribe((event:NavigationStart) => {
                console.log(event);
                this.toggleSearchBox(event.url);
        });
    }

    toggleSearchBox(currenturl: string){
        if(currenturl == "/home" || currenturl == "/"){
            this.ishome = true;
        }
        else{
            this.ishome = false;
        }
    }

    logout() {
        this._authenticationService.logout()
            .subscribe(null, null, () => {
                localStorage.removeItem('currentUser');
        })
    }
   
}